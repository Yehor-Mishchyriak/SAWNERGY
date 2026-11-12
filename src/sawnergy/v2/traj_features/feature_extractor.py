"""Subclass-defined trajectory features, execution contexts, and archive publication."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from inspect import isfunction
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType, MethodType

import icontract
import numpy as np

from sawnergy.sawnergy_util import ArrayStorage
from sawnergy import __version__, sawnergy_util
from sawnergy._archive_schema import validate_archive_schema
from sawnergy.rin import rin_util


def _positive_int(value) -> bool:
    """Check for a positive Python integer, excluding bools."""
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
    dimensions, including scalar ``()``. Window outputs prepend an axis of
    ``len(ctx.windows)`` samples; trajectory outputs have the declared sample shape.
    ``dtype`` is an explicit real numeric/bool dtype. ``units`` and ``statistic``
    are nonblank descriptions, with ``"1"`` for dimensionless quantities.
    Methods return finite, unmasked ndarrays of the exact declared dtype.
    Both pairwise axes follow ``ctx.residue_ids``. The method implements its
    physical definition, coordinate processing, and declared reduction.

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
        """Attach the validated declarations and return the original method."""
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
    Optional ``rows=(start, stop)`` exposes a valid half-open block interval,
    with read indices relative to that interval. The caller supplies its bounds.
    """

    @icontract.require(lambda storage: isinstance(storage, ArrayStorage), error=TypeError, enabled=True)
    @icontract.require(
        lambda block_name: isinstance(block_name, str) and bool(block_name),
        error=lambda: ValueError("block_name must be a nonempty string."), enabled=True)
    @icontract.require(
        lambda rows: rows is None or (isinstance(rows, tuple) and len(rows) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in rows)
            and rows[0] <= rows[1]), error=lambda: ValueError("rows must be a nonnegative (start, stop) interval."),
        enabled=True)
    def __init__(self, storage: ArrayStorage, block_name: str, *, rows: tuple[int, int] | None = None):
        """Borrow a block, optionally restricting reads to caller-supplied row bounds."""
        self._storage = storage
        self._block_name = block_name
        self._rows = rows

    def read(self, ids: int | slice | tuple[int, ...] | None = None) -> np.ndarray:
        """Read copied rows relative to this view; None selects all, an int drops the row axis."""
        if self._rows is not None:
            indices = range(*self._rows)
            if ids is None:
                ids = slice(*self._rows)
            elif isinstance(ids, int):
                ids = indices[ids]
            elif isinstance(ids, slice):
                ids = tuple(indices[ids])
            else:
                ids = tuple(indices[index] for index in ids)
        return self._storage.read(self._block_name, ids)

    def block_iter(self, *, step: int = 1) -> Iterator[np.ndarray]:
        """Yield copied batches of at most ``step`` rows through ArrayStorage."""
        if self._rows is None:
            yield from self._storage.block_iter(self._block_name, step=step)
        else:
            if not _positive_int(step):
                raise ValueError("step must be a positive integer.")
            start, stop = self._rows
            for offset in range(start, stop, step):
                yield self._storage.read(self._block_name, slice(offset, min(offset + step, stop)))


@dataclass(frozen=True, kw_only=True)
class ExtractionContext:
    """Snapshot the inputs and dependency readers for one feature call.

    Paths become resolved Path objects. Residue pairs, atom memberships,
    source frames, and windows become nested tuples. Dependencies become a
    copied read-only mapping of declared feature names to block readers.

    ``residue_ids`` holds 1-based ``(molecule_id, topology_global_residue_id)``
    pairs in residue order. ``atom_ids[i]`` holds the ordered, 1-based
    topology-global atom IDs belonging to ``residue_ids[i]``.

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
        """Freeze normalized input containers and validate identity/frame structure."""
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
    """Extract subclass-defined features into a validated trajectory archive.

    Subclasses mark methods accepting ``(self, ctx)`` with ``@feature(...)``.
    ``featurize()`` prepares inputs, executes those methods in discovery order,
    and publishes their outputs. Feature methods own CPPTRAJ calls and reductions
    and restrict all computation to the context's selected atoms and frames.
    """

    @icontract.require(lambda batch_size: _positive_int(batch_size), error=ValueError, enabled=True)
    @icontract.require(
        lambda compression_level: isinstance(compression_level, int) and not isinstance(compression_level, bool)
        and 0 <= compression_level <= 9,
        error=lambda: ValueError("compression_level must be an integer from 0 through 9."), enabled=True)
    def featurize(self, topology_file: Path | str, trajectory_file: Path | str, cpptraj_path: Path | str,
                  selection: Mapping[int, Sequence[int]], output_path: Path | str, *,
                  frame_range: tuple[int, int] | None = None, frame_stride: int = 1, window_size: int = 1,
                  batch_size: int = 1, compression_level: int = 3) -> str:
        """Extract declared features into a validated trajectory-feature ZIP archive.

        Sources must be existing files and ``cpptraj_path`` a verified executable.
        ``selection={molecule_id: [residue_ids...]}`` selects complete residues
        using 1-based, topology-global residue IDs, ordered by topology internally.
        ``frame_range`` is a 1-based inclusive pair or None for the whole file;
        ``frame_stride`` selects every nth frame within that range.

        Complete each feature before the next. Window dependencies expose the
        matching batch to window consumers and the full block to trajectory
        consumers. Trajectory dependencies expose one stored sample.

        ``window_size`` counts selected frames in consecutive, nonoverlapping
        windows, retaining the final short window. ``batch_size`` counts windows
        per method call. ``frame_stride``, ``window_size``, and ``batch_size``
        must be positive integers.
        Trajectory methods receive all selected frames. ``output_path`` names a
        ZIP file and ``compression_level`` is an integer from 0 through 9.

        Validate each output before writing and the finished store before
        publication. Failures propagate through ArrayStorage's cleanup context.
        Pre-publication failures preserve any existing destination; cleanup errors
        after publication may propagate with the new archive already in place.
        Return the resolved destination path after successful publication.
        """
        output = Path(output_path).resolve()
        if output.suffix.lower() != ".zip":
            raise ValueError("output_path must name a .zip file.")
        if output.is_dir():
            raise IsADirectoryError(output)
        methods = self._feature_methods()
        if not methods:
            raise ValueError("At least one feature is required.")
        ctx = self._prepare_inputs(topology_file, trajectory_file, cpptraj_path, selection,
                                   frame_range=frame_range, frame_stride=frame_stride, window_size=window_size)
        attrs, identities = self._archive_layout(ctx, methods, frame_range, frame_stride, window_size)
        declarations = {entry["name"]: entry for entry in attrs["features"]}
        with ArrayStorage.compress_and_cleanup(output, compression_level) as storage:
            for key, value in json.loads(json.dumps(attrs)).items():
                storage.add_attr(key, value)
            for key, values in identities.items():
                storage.write(values, key, arrays_per_chunk=4096)
            for name, method in methods.items():
                declaration = declarations[name]
                windowed = declaration["scope"] == "window"
                contexts = self._window_batches(ctx, batch_size=batch_size) if windowed else (ctx,)
                offset = 0
                for current in contexts:
                    dependencies = {}
                    for required in declaration["requires"]:
                        dependency = declarations[required]
                        rows = ((offset, offset + len(current.windows))
                                if windowed and dependency["scope"] == "window" else None)
                        dependencies[required] = DependencyReader(storage, dependency["block"], rows=rows)
                    current = replace(current, dependencies=dependencies)
                    values = method(current)
                    self._validate_output(method, values, current)
                    values = values.astype(declaration["dtype"], copy=False)
                    storage.write(values if windowed else values[None], declaration["block"],
                                  arrays_per_chunk=batch_size if windowed else 1)
                    offset += len(current.windows)
                    del values
            self._validate_archive(storage, attrs, identities, batch_size=batch_size)
        return str(output)

    def _archive_layout(self, ctx, methods, frame_range, frame_stride, window_size):
        """Build independent expected declarations and identity arrays from prepared inputs."""
        features = [dict(name=name, block=f"feature_{index}", kind=method.feature_kind,
                         scope=method.feature_scope, shape=list(method.feature_shape),
                         dtype=method.feature_dtype.name, units=method.feature_units,
                         statistic=method.feature_statistic, requires=list(method.feature_requires))
                    for index, (name, method) in enumerate(methods.items())]
        attrs = dict(producer_version=__version__, schema_name="traj_features", schema_version=1,
                     topology_file=str(ctx.topology_file), trajectory_file=str(ctx.trajectory_file),
                     cpptraj=str(ctx.cpptraj), frame_range=list(frame_range) if frame_range is not None else None,
                     frame_stride=frame_stride, window_size=window_size, features=features)
        identities = dict(residue_ids=np.asarray(ctx.residue_ids, dtype="int64"),
                          atom_ids=np.asarray([atom for ids in ctx.atom_ids for atom in ids], dtype="int64"),
                          atom_offsets=np.cumsum([0] + [len(ids) for ids in ctx.atom_ids], dtype="int64"),
                          frame_ids=np.asarray(ctx.frame_ids, dtype="int64"),
                          window_offsets=np.cumsum([0] + [len(window) for window in ctx.windows], dtype="int64"))
        return attrs, identities

    def _validate_archive(self, storage, attrs, identities, *, batch_size):
        """Reread stored metadata and values against the independently prepared contract."""
        validate_archive_schema(storage.root.attrs, schema_name="traj_features", schema_version=1)
        try:
            actual = {key: storage.get_attr(key) for key in attrs}
            if not isinstance(actual["features"], list) or len(actual["features"]) != len(attrs["features"]):
                raise ValueError("Invalid feature manifest.")
            for entry, expected in zip(actual["features"], attrs["features"]):
                if not isinstance(entry, dict):
                    raise ValueError("Invalid feature declaration.")
                if entry["dtype"] != expected["dtype"]:
                    raise TypeError("Feature dtype declaration differs from the extraction contract.")
            actual["features"] = [{key: entry[key] for key in expected}
                                  for entry, expected in zip(actual["features"], attrs["features"])]
            if json.dumps(actual, sort_keys=True) != json.dumps(attrs, sort_keys=True):
                raise ValueError("Archive metadata differs from the extraction contract.")
            maps = {key: storage.get_attr(key) for key in (
                "array_chunk_size_in_block", "array_shape_in_block", "array_dtype_in_block")}
        except KeyError as error:
            raise ValueError(f"Missing archive metadata: {error}") from error

        shapes = {key: values.shape for key, values in identities.items()}
        dtypes = {key: "int64" for key in identities}
        for entry in attrs["features"]:
            shape = ((len(identities["residue_ids"]),) * 2 if entry["kind"] == "pairwise" else tuple(entry["shape"]))
            rows = len(identities["window_offsets"]) - 1 if entry["scope"] == "window" else 1
            shapes[entry["block"]] = (rows, *shape)
            dtypes[entry["block"]] = entry["dtype"]
        if set(storage.root.keys()) != set(shapes) or set(storage.list_blocks()) != set(shapes):
            raise ValueError("Archive datasets differ from the extraction contract.")
        if any(not isinstance(mapping, dict) or set(mapping) != set(shapes) for mapping in maps.values()):
            raise ValueError("Invalid storage metadata maps.")
        for key, shape in shapes.items():
            block = storage.root[key]
            if block.shape != shape:
                raise ValueError(f"Invalid stored shape for {key!r}.")
            cached_dtype = maps["array_dtype_in_block"][key]
            if (block.dtype.name != dtypes[key] or not isinstance(cached_dtype, str)
                    or np.dtype(cached_dtype).name != dtypes[key]):
                raise TypeError(f"Invalid stored dtype for {key!r}.")
            if json.dumps(maps["array_shape_in_block"][key]) != json.dumps(list(shape[1:])):
                raise ValueError(f"Invalid stored item shape for {key!r}.")
            chunk = maps["array_chunk_size_in_block"][key]
            if type(chunk) is not int or chunk != block.chunks[0]:
                raise ValueError(f"Invalid stored chunk length for {key!r}.")
            offset = 0
            for values in storage.block_iter(key, step=4096 if key in identities else batch_size):
                if key in identities:
                    if not np.array_equal(values, identities[key][offset:offset + len(values)]):
                        raise ValueError(f"Stored identities differ for {key!r}.")
                elif not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite stored feature values in {key!r}.")
                offset += len(values)

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
        Identity alignment is the feature author's responsibility.
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
