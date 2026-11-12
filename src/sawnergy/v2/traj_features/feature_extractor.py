"""Feature metadata, execution inputs, and ordered discovery for trajectory extractors."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from inspect import isfunction
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType, MethodType

import icontract
import numpy as np

from sawnergy.sawnergy_util import ArrayStorage
from sawnergy import sawnergy_util
from sawnergy.rin import rin_util


def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _ordered_ids(values: tuple[int, ...]) -> bool:
    """Check nonempty, positive, strictly increasing integer IDs, excluding bools."""
    return (bool(values)
            and all(_positive_int(value) for value in values)
            and all(left < right for left, right in zip(values, values[1:])))


@icontract.require(lambda kind: kind in ("pairwise", "global"), error=ValueError, enabled=True)
@icontract.require(lambda scope: scope in ("window", "trajectory"), error=ValueError, enabled=True)
@icontract.require(
    lambda kind, shape: isinstance(shape, tuple) and (
        shape == ("N", "N") if kind == "pairwise" else
        all(isinstance(size, int) and not isinstance(size, bool) and size >= 0 for size in shape)),
    error=lambda: ValueError("shape must be ('N', 'N') for pairwise features "
                             "or a tuple of nonnegative integers for global features."),
    enabled=True)
@icontract.require(
    lambda units, statistic: all(isinstance(value, str) and bool(value.strip()) for value in (units, statistic)),
    error=lambda: ValueError("units and statistic must be nonblank strings."), enabled=True)
@icontract.require(
    lambda requires: isinstance(requires, tuple)
    and all(isinstance(name, str) and name != "" for name in requires)
    and len(requires) == len(set(requires)),
    # Avoid reevaluating guarded expressions while formatting contract errors.
    error=lambda: ValueError("requires must be a tuple of distinct nonempty feature names."),
    enabled=True)
def feature(*, kind: str, scope: str, shape: tuple[int | str, ...], dtype: np.dtype | str | type,
            units: str, statistic: str, requires: tuple[str, ...] = ()):
    """Attach feature metadata to an ordinary instance method.

    ``kind`` accepts ``"pairwise"`` for residue-pair features or ``"global"``
    for features describing the selected system. ``scope`` accepts ``"window"``
    or ``"trajectory"``. All four combinations are allowed.

    ``shape`` describes one sample. Pairwise features declare ``("N", "N")``,
    with N selected residues. Global features declare fixed nonnegative integer
    dimensions, including scalar ``()``. Window outputs prepend a batch axis.
    ``dtype`` is an explicit real numeric/bool dtype. ``units`` and ``statistic``
    are nonblank descriptions, with ``"1"`` for dimensionless quantities.

    ``requires`` is a tuple of distinct nonempty feature names, defaulting to
    an empty tuple. Each dependency must precede this feature in the final
    discovery order, after inheritance and overrides have been resolved.

    The returned decorator stores each declaration as ``feature_<name>``
    on the original function and returns it. ``feature_dtype`` is a NumPy dtype.
    Decoration happens at class definition.
    Python supplies ``self`` when the method is later called on an instance.

    Invalid dtype raises TypeError. Other invalid declarations raise ValueError
    when ``feature(...)`` is called. Checks remain active under ``python -O``.

    Example usage within a subclass

        @feature(kind="pairwise", scope="window", shape=("N", "N"),
                 dtype="float32", units="angstrom", statistic="mean per-frame distance")
        def distances(self, ctx):
            ...
    """
    if dtype is None:
        raise TypeError("dtype must be explicitly declared.")
    try:
        dtype = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise TypeError("dtype must be a real numeric or boolean NumPy dtype.") from error
    if dtype.kind not in "biuf":
        raise TypeError("dtype must be a real numeric or boolean NumPy dtype.")

    def decorate(method):
        method.feature_kind = kind
        method.feature_scope = scope
        method.feature_shape = shape
        method.feature_dtype = dtype
        method.feature_units = units
        method.feature_statistic = statistic
        method.feature_requires = requires
        return method

    return decorate


class DependencyReader:
    """Read one dependency block from a store owned by the extraction caller.

    ``read(ids)`` selects zero-based stored rows with the same indexing as
    ``ArrayStorage.read``. ``block_iter(step=...)`` streams row batches.
    Both return independent ndarray copies. Access requires the owner to keep
    the borrowed store open for the duration of the feature call.
    """

    @icontract.require(lambda storage: isinstance(storage, ArrayStorage), error=TypeError, enabled=True)
    @icontract.require(
        lambda block_name: isinstance(block_name, str) and bool(block_name),
        error=lambda: ValueError("block_name must be a nonempty string."), enabled=True)
    def __init__(self, storage: ArrayStorage, block_name: str):
        self._storage = storage
        self._block_name = block_name

    def read(self, ids: int | slice | tuple[int, ...] | None = None) -> np.ndarray:
        """Read selected rows as a copy, propagating storage errors."""
        return self._storage.read(self._block_name, ids)

    def block_iter(self, *, step: int = 1) -> Iterator[np.ndarray]:
        """Yield copied batches of at most ``step`` rows through ArrayStorage."""
        return self._storage.block_iter(self._block_name, step=step)


@dataclass(frozen=True, kw_only=True)
class ExtractionContext:
    """Snapshot the inputs and dependency readers for one feature call.

    Paths become resolved Path objects. Residue pairs, atom memberships,
    source frames, and windows become nested tuples. Dependencies become a
    copied read-only mapping of declared feature names to block readers.

    ``frame_ids`` contains the explicit 1-based source frames for this call.
    Optional ``windows`` group those frames and may overlap. Structural
    violations raise ValueError, including under ``python -O``.

    The caller validates source files and selects complete residues by molecule/residue ID,
    and supplies completed dependencies matching the feature's ``requires``.
    Context construction normalizes and checks the supplied structure.
    """

    cpptraj: Path | str
    topology_file: Path | str
    trajectory_file: Path | str
    residue_ids: tuple[tuple[int, int], ...]
    atom_ids: tuple[tuple[int, ...], ...]
    frame_ids: tuple[int, ...]
    windows: tuple[tuple[int, ...], ...] = ()
    dependencies: Mapping[str, DependencyReader] = field(default_factory=dict)

    @icontract.ensure(lambda self: isinstance(self.dependencies, MappingProxyType), enabled=True)
    def __post_init__(self):
        for name in ("cpptraj", "topology_file", "trajectory_file"):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        object.__setattr__(self, "residue_ids", tuple(tuple(pair) for pair in self.residue_ids))
        object.__setattr__(self, "atom_ids", tuple(tuple(ids) for ids in self.atom_ids))
        object.__setattr__(self, "frame_ids", tuple(self.frame_ids))
        object.__setattr__(self, "windows", tuple(tuple(window) for window in self.windows))
        object.__setattr__(self, "dependencies", MappingProxyType(dict(self.dependencies)))

        if not self.residue_ids or any(
            len(pair) != 2 or any(not _positive_int(value) for value in pair) for pair in self.residue_ids
        ):
            raise ValueError("residue_ids must contain positive integer (molecule_id, residue_id) pairs.")
        if not _ordered_ids(tuple(pair[1] for pair in self.residue_ids)):
            raise ValueError("residue_ids must be in strictly increasing topology residue order.")
        if len(self.atom_ids) != len(self.residue_ids) or not all(_ordered_ids(ids) for ids in self.atom_ids):
            raise ValueError("atom_ids must contain nonempty ordered atom IDs for each residue.")
        atoms = [atom for ids in self.atom_ids for atom in ids]
        if len(set(atoms)) != len(atoms):
            raise ValueError("atom_ids must be disjoint across residues.")
        if not _ordered_ids(self.frame_ids):
            raise ValueError("frame_ids must be nonempty, positive, and strictly increasing.")
        if any(not _ordered_ids(window) for window in self.windows):
            raise ValueError("Each window must contain nonempty, positive, strictly increasing frame IDs.")
        if self.windows and set(self.frame_ids) != {frame for window in self.windows for frame in window}:
            raise ValueError("The union of windows must equal frame_ids.")
        if any(not isinstance(name, str) or not name or not isinstance(reader, DependencyReader)
               for name, reader in self.dependencies.items()):
            raise ValueError("dependencies must map nonempty feature names to DependencyReader objects.")

    @property
    def selection_mask(self) -> str:
        """CPPTRAJ mask for the fixed, topology-global residue IDs in this context."""
        return ":" + ",".join(str(residue) for _, residue in self.residue_ids)


class FeatureExtractor:
    """Base class providing discovery of subclass-defined feature methods.

    Subclasses mark methods accepting ``(self, ctx)`` with ``@feature(...)``.
    The private ``_feature_methods()`` helper returns those methods bound to
    the current instance, ready to be called with ``ctx``.
    """

    @icontract.require(
        lambda frame_stride, window_size: all(_positive_int(value) for value in (frame_stride, window_size)),
        error=lambda: ValueError("frame_stride and window_size must be positive integers."), enabled=True)
    @icontract.require(
        lambda frame_range: frame_range is None or (
            isinstance(frame_range, tuple) and len(frame_range) == 2
            and all(_positive_int(value) for value in frame_range) and frame_range[0] <= frame_range[1]),
        error=lambda: ValueError("frame_range must be an ordered, positive (first, last) tuple."), enabled=True)
    def _prepare_inputs(self, topology_file: Path | str, trajectory_file: Path | str,
                        cpptraj_path: Path | str, selection: Mapping[int, Sequence[int]], *,
                        frame_range: tuple[int, int] | None = None, frame_stride: int = 1,
                        window_size: int = 1) -> ExtractionContext:
        """Validate sources and prepare fixed identities, selected frames, and windows.

        ``selection`` maps molecule IDs to explicit topology-global residue IDs.
        Every pair must exist. Complete atom memberships are read from CPPTRAJ
        and ordered by topology residue ID. The context derives its numeric mask.

        Frame bounds are 1-based and inclusive. Windows partition selected frames,
        retaining the final short window. Trajectory features use the full context.
        """
        if not isinstance(selection, Mapping):
            raise TypeError("selection must map molecule IDs to residue IDs.")
        if not selection:
            raise ValueError("selection must be nonempty.")
        requested = {}
        for molecule, residues in selection.items():
            ids = tuple(residues)
            if (not _positive_int(molecule) or not ids or not all(_positive_int(value) for value in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError("selection requires positive molecule IDs and distinct positive residue IDs.")
            requested[molecule] = ids

        topology_file, trajectory_file, cpptraj = (
            Path(path).resolve() for path in (topology_file, trajectory_file, cpptraj_path))
        for path in (topology_file, trajectory_file, cpptraj):
            if not path.is_file():
                raise FileNotFoundError(f"Input must be an existing file: {path}")
        if not os.access(cpptraj, os.X_OK):
            raise PermissionError(f"CPPTRAJ is not executable: {cpptraj}")
        verified = Path(rin_util.locate_cpptraj(explicit=cpptraj, verify=True)).resolve()
        if verified != cpptraj:
            raise rin_util.CpptrajNotFound([cpptraj])

        with TemporaryDirectory(prefix="sawnergy_input_") as directory:
            # CPPTRAJ can split filenames containing spaces even when quoted.
            topology = Path(directory) / f"topology{topology_file.suffix}"
            trajectory = Path(directory) / f"trajectory{trajectory_file.suffix}"
            topology.symlink_to(topology_file)
            trajectory.symlink_to(trajectory_file)
            raw = rin_util.run_cpptraj(str(cpptraj), argv=["-p", str(topology), "-y", str(trajectory), "-tl"])
            try:
                total_frames = int(raw.strip().removeprefix("Frames: "))
            except ValueError as error:
                raise RuntimeError("CPPTRAJ returned an invalid trajectory frame count.") from error
            first, last = frame_range if frame_range is not None else (1, total_frames)
            if total_frames <= 0 or last > total_frames:
                raise ValueError(f"frame_range must lie within [1, {total_frames}].")
            frames = tuple(range(first, last + 1, frame_stride))
            windows = tuple(sawnergy_util.batches_of(frames, window_size, out_as=tuple))

            table = Path(directory) / "atoms.dat"
            script = rin_util.CpptrajScript((
                f'trajin "{trajectory}" {first} {first}', f'mask :* out "{table}"')).render()
            rin_util.run_cpptraj(str(cpptraj), argv=["-p", str(topology)], script=script)
            composition = rin_util.CpptrajMaskParser.hierarchize_molecular_composition(str(table))
        residue_ids = tuple(sorted(((molecule, residue) for molecule, ids in requested.items()
                                    for residue in ids), key=lambda pair: pair[1]))
        atom_ids = []
        for molecule, residue in residue_ids:
            if residue not in composition.get(molecule, {}):
                raise ValueError(f"Unknown molecule/residue pair ({molecule}, {residue}).")
            atom_ids.append(tuple(sorted(composition[molecule][residue])))
        return ExtractionContext(cpptraj=cpptraj, topology_file=topology_file, trajectory_file=trajectory_file,
                                 residue_ids=residue_ids, atom_ids=tuple(atom_ids), frame_ids=frames, windows=windows)

    @icontract.require(lambda batch_size: _positive_int(batch_size), error=ValueError, enabled=True)
    @icontract.require(lambda ctx: bool(ctx.windows), error=ValueError, enabled=True)
    def _window_batches(self, ctx: ExtractionContext, *, batch_size: int = 1) -> Iterator[ExtractionContext]:
        """Yield contexts for consecutive window batches with the same fixed selection."""
        for windows in sawnergy_util.batches_of(ctx.windows, batch_size, out_as=tuple):
            frames = tuple(sorted({frame for window in windows for frame in window}))
            yield replace(ctx, frame_ids=frames, windows=windows)

    @icontract.ensure(lambda self, result: all(method.__self__ is self for method in result.values()),
                      enabled=True)
    def _feature_methods(self) -> dict[str, MethodType]:
        """Return a fresh ordered mapping from feature names to bound methods.

        Class dictionaries are visited in reverse method resolution order
        (MRO), giving subclass definitions precedence over base definitions.
        Only ordinary instance functions with both feature metadata attributes
        are included. Function and property bodies remain unevaluated.

        Inherited features precede subclass additions. A decorated override
        retains its position. An undecorated override removes the inherited
        feature, and decorating that name in a later subclass appends it anew.

        After resolving inheritance, each dependency must name an earlier
        feature. Missing, self, or later dependencies raise ValueError naming
        the dependent feature and unavailable dependencies, even under ``python -O``.

        Each returned method is bound to this instance, so ``method(ctx)``
        supplies ``self`` automatically. The postcondition checks this binding
        even under ``python -O``. Editing the returned dictionary affects only
        that discovery result.
        """
        methods = {}
        # Visit base classes first so subclasses determine the final definition.
        for cls in reversed(type(self).__mro__):
            for name, method in vars(cls).items():
                if (isfunction(method) and hasattr(method, "feature_kind")
                        and hasattr(method, "feature_scope")):
                    methods[name] = method
                else:
                    # An undecorated override removes an inherited feature.
                    methods.pop(name, None)

        # Validate the final order after inherited features have been replaced or removed.
        earlier = set()
        for name, method in methods.items():
            unavailable = [required for required in method.feature_requires if required not in earlier]
            if unavailable:
                raise ValueError(
                    f"Feature {name!r} requires {unavailable!r} to appear earlier in discovery order.")
            earlier.add(name)

        # Bind each function to self so callers pass only ctx.
        return {name: method.__get__(self, type(self)) for name, method in methods.items()}

    def _validate_output(self, method: MethodType, values: np.ndarray, ctx: ExtractionContext) -> None:
        """Check declared shape/dtype and finite, unmasked ndarray values.

        Window features require one result per context window. Pairwise axes
        follow ``ctx.residue_ids`` by the feature implementation's contract.
        Backend identity alignment is checked by concrete features and tests.
        """
        name = method.__name__
        if not isinstance(values, np.ndarray) or np.ma.isMaskedArray(values):
            raise TypeError(f"Feature {name!r} must return an unmasked ndarray.")
        if values.dtype != method.feature_dtype:
            raise TypeError(f"Feature {name!r} dtype {values.dtype} != {method.feature_dtype}.")
        shape = ((len(ctx.residue_ids),) * 2 if method.feature_kind == "pairwise" else method.feature_shape)
        if method.feature_scope == "window":
            if not ctx.windows:
                raise ValueError(f"Window feature {name!r} requires context windows.")
            shape = (len(ctx.windows), *shape)
        if values.shape != shape:
            raise ValueError(f"Feature {name!r} shape {values.shape} != {shape}.")
        if not np.isfinite(values).all():
            raise ValueError(f"Feature {name!r} contains NaN or infinite values.")
