from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List

import numpy as np
import pytest

from sawnergy.rin import rin_builder, rin_util
from sawnergy.walks import walker, walker_util
from sawnergy.embedding import embedder as embedder_module
from sawnergy.visual import visualizer as visualizer_module
from sawnergy import sawnergy_util


# ---------------------------------------------------------------------------
# Synthetic cpptraj fixtures
# ---------------------------------------------------------------------------

RESIDUE_COUNT = 3
FRAME_COUNT = 2


def _format_square_block(matrix: np.ndarray) -> str:
    rows = [" ".join(f"{val:.6f}" for val in row) for row in matrix]
    return "\n".join(rows)


def _pairwise_output(matrix: np.ndarray) -> str:
    zeros = np.zeros_like(matrix)
    return (
        "[printdata PW[EMAP] square2d noheader]\n"
        f"{_format_square_block(zeros)}\n"
        "[printdata PW[VMAP] square2d noheader]\n"
        f"{_format_square_block(matrix)}\n"
    )


def _com_output(frame_indices: Iterable[int], coords_lookup: Dict[int, np.ndarray]) -> str:
    lines = [
        "Some header line",
        f"COMZ{RESIDUE_COUNT}  dataset",
    ]
    for frame in frame_indices:
        coords = coords_lookup[frame]
        flat_values = np.concatenate(
            [coords[:, 0], coords[:, 1], coords[:, 2]]
        )
        flat = " ".join(f"{val:.6f}" for val in flat_values)
        lines.append(f"{frame} {flat}")
    lines.append("[quit]")
    return "\n".join(lines) + "\n"


PAIRWISE_MATRICES: Dict[int, np.ndarray] = {
    1: np.array(
        [
            [0.0, -2.0, 1.5],
            [-2.0, 0.0, 3.0],
            [1.5, -1.5, 0.0],
        ],
        dtype=np.float32,
    ),
    2: np.array(
        [
            [0.0, -1.0, 2.5],
            [-1.0, 0.0, 1.0],
            [2.5, -0.5, 0.0],
        ],
        dtype=np.float32,
    ),
}

COM_COORDS: Dict[int, np.ndarray] = {
    1: np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
    2: np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32),
}


def compute_processed_channels(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residue = matrix.copy().astype(np.float32)
    attr = np.where(residue <= 0, -residue, 0.0).astype(np.float32)
    rep = np.where(residue > 0, residue, 0.0).astype(np.float32)

    attr_threshold = np.quantile(attr, 1.0, axis=1, keepdims=True)
    attr = np.where(attr < attr_threshold, 0.0, attr)
    rep_threshold = np.quantile(rep, 1.0, axis=1, keepdims=True)
    rep = np.where(rep < rep_threshold, 0.0, rep)

    np.fill_diagonal(attr, 0.0)
    np.fill_diagonal(rep, 0.0)

    attr = 0.5 * (attr + attr.T)
    rep = 0.5 * (rep + rep.T)

    row_sums = attr.sum(axis=1, keepdims=True)
    normalized_attr = np.divide(
        attr,
        np.clip(row_sums, 1e-12, None),
        out=np.zeros_like(attr),
        where=row_sums > 0,
    )
    return attr, rep, normalized_attr

COMPOSITION_TEXT = "\n".join(
    [
        "[AtNum] [Rnum] [Mnum]",
        "1 1 1",
        "2 2 1",
        "3 3 1",
        "",
    ]
)


@pytest.fixture
def patched_cpptraj(monkeypatch, tmp_path: Path):
    """Patch cpptraj helpers so the builder operates on synthetic data."""

    def fake_locate(explicit=None, verify=True):
        return "/usr/bin/cpptraj"

    monkeypatch.setattr(rin_util, "locate_cpptraj", fake_locate)

    def fake_run_cpptraj(
        executable,
        argv=None,
        script=None,
        env=None,
        timeout=None,
    ):
        if argv and "-tl" in argv:
            return f"Frames: {FRAME_COUNT}"

        if script is None:
            return ""

        if "mask :*" in script:
            match = re.search(r"mask :\*.*out\s+(\S+)", script)
            if not match:
                raise RuntimeError("Could not locate output path for mask command")
            Path(match.group(1)).write_text(COMPOSITION_TEXT)
            return ""

        if "printdata PW[EMAP]" in script:
            frame_match = re.search(r"trajin\s+\S+\s+(\d+)\s+(\d+)", script)
            if not frame_match:
                raise RuntimeError("Could not infer frame range for pairwise output")
            start = int(frame_match.group(1))
            matrix = PAIRWISE_MATRICES[start]
            return _pairwise_output(matrix)

        if "printdata COMX" in script:
            frame_match = re.search(r"trajin\s+\S+\s+(\d+)\s+(\d+)", script)
            if not frame_match:
                raise RuntimeError("Could not infer frame range for COM output")
            start = int(frame_match.group(1))
            end = int(frame_match.group(2))
            return _com_output(range(start, end + 1), COM_COORDS)

        return ""

    monkeypatch.setattr(rin_util, "run_cpptraj", fake_run_cpptraj)

    return tmp_path


@pytest.fixture
def rin_archive_path(patched_cpptraj: Path) -> Path:
    output_path = patched_cpptraj / "synthetic_rin.zip"
    builder = rin_builder.RINBuilder(cpptraj_path="cpptraj")
    builder.build_rin(
        topology_file="top.prmtop",
        trajectory_file="traj.nc",
        molecule_of_interest=1,
        frame_range=(1, FRAME_COUNT),
        frame_batch_size=1,
        prune_low_energies_frac=1.0,
        output_path=output_path,
        keep_prenormalized_energies=True,
        include_attractive=True,
        include_repulsive=False,
        compression_level=0,
        num_matrices_in_compressed_blocks=1,
    )
    return output_path


class _SharedArrayStub:
    def __init__(self, array: np.ndarray):
        self._array = np.array(array, copy=True)
        self.shape = self._array.shape
        self.dtype = self._array.dtype
        self.name = "stub"

    @property
    def array(self):
        return self._array

    def __getitem__(self, item):
        return self._array[item]

    def view(self, *_, **__):
        return self._array

    def close(self):
        return None

    def unlink(self):
        return None

    @classmethod
    def create(cls, shape, dtype, *, from_array=None, name=None):
        if from_array is None:
            buf = np.zeros(shape, dtype=dtype)
        else:
            buf = np.array(from_array, dtype=dtype, copy=True)
        return cls(buf)


@pytest.fixture
def patched_shared_ndarray(monkeypatch):
    monkeypatch.setattr(walker_util, "SharedNDArray", _SharedArrayStub)
    return _SharedArrayStub


@pytest.fixture
def walks_archive_path(
    rin_archive_path: Path,
    tmp_path: Path,
    patched_shared_ndarray,
) -> Path:
    walker_obj = walker.Walker(rin_archive_path, seed=123)
    out_path = tmp_path / "synthetic_walks.zip"
    walker_obj.sample_walks(
        walk_length=2,
        walks_per_node=1,
        saw_frac=0.0,
        include_attractive=True,
        include_repulsive=False,
        time_aware=False,
        output_path=out_path,
        in_parallel=False,
        compression_level=0,
        num_walk_matrices_in_compressed_blocks=1,
    )
    walker_obj.close()
    return out_path


# ---------------------------------------------------------------------------
# Stub SGNS implementation for embedding tests
# ---------------------------------------------------------------------------

class _StubSGNS:
    call_log: List[int] = []

    def __init__(self, V: int, D: int, *, seed: int | None = None, **_):
        self.V = V
        self.D = D
        self.seed = 0 if seed is None else int(seed)
        self._embeddings = np.zeros((V, D), dtype=np.float32)

    def fit(
        self,
        centers: np.ndarray,
        contexts: np.ndarray,
        num_epochs: int,
        batch_size: int,
        num_negative_samples: int,
        noise_dist: np.ndarray,
        shuffle_data: bool,
        lr_step_per_batch: bool,
    ):
        rng = np.random.default_rng(self.seed)
        base = rng.random()
        values = np.linspace(0.0, 1.0, self.D, dtype=np.float32)
        self._embeddings = (values + base).repeat(self.V).reshape(self.V, self.D)
        self.call_log.append(self.seed)

    @property
    def embeddings(self) -> np.ndarray:
        return self._embeddings


@pytest.fixture
def patched_sgns(monkeypatch):
    monkeypatch.setattr(
        embedder_module.Embedder,
        "_get_SGNS_constructor_from",
        staticmethod(lambda base: _StubSGNS),
    )
    return _StubSGNS


@pytest.fixture
def embeddings_archive_path(
    walks_archive_path: Path,
    patched_sgns,
    tmp_path: Path,
) -> Path:
    _StubSGNS.call_log.clear()

    emb = embedder_module.Embedder(walks_archive_path, base="torch", seed=999)
    out_path = tmp_path / "synthetic_embeddings.zip"
    emb.embed_all(
        RIN_type="attr",
        using="RW",
        window_size=1,
        num_negative_samples=1,
        num_epochs=1,
        batch_size=4,
        shuffle_data=False,
        dimensionality=2,
        alpha=0.75,
        output_path=out_path,
        sgns_kwargs={},
    )
    return out_path


# ---------------------------------------------------------------------------
# Matplotlib stubs for Visualizer tests
# ---------------------------------------------------------------------------

class _DummyScatter:
    def __init__(self, *_, **__):
        self._offsets3d = ([], [], [])

    def set_facecolors(self, *_):
        return None


class _DummyAxes:
    def __init__(self):
        self._collections = []

    def add_collection3d(self, collection):
        self._collections.append(collection)

    def set_autoscale_on(self, *_):
        pass

    def view_init(self, *_):
        pass

    def set_axis_off(self):
        pass

    def text(self, *args, **kwargs):
        return None

    def text2D(self, *args, **kwargs):
        return None

    def set_xlim(self, *_):
        pass

    def set_ylim(self, *_):
        pass

    def set_zlim(self, *_):
        pass

    def set_box_aspect(self, *_):
        pass

    def scatter(self, *_, **__):
        return _DummyScatter()


class _DummyFigure:
    def __init__(self, *_, **__):
        self.patch = SimpleNamespace(set_facecolor=lambda *_: None)

    def subplots_adjust(self, *_, **__):
        pass

    def add_subplot(self, *_ , **__):
        return _DummyAxes()


class _DummyLineCollection:
    def __init__(self, *_, **__):
        self._segments = None
        self._colors = None
        self._alpha = None

    def set_segments(self, segs):
        self._segments = np.array(segs, copy=True)

    def set_colors(self, colors):
        self._colors = np.array(colors, copy=True)

    def set_alpha(self, alpha):
        self._alpha = alpha


class _DummyNormalize:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, value):
        return value


class _DummyCMap:
    def __call__(self, values):
        arr = np.asarray(values)
        if arr.ndim == 0:
            arr = arr[None]
        return np.tile(np.array([1.0, 0.0, 0.0, 1.0]), (len(arr), 1))


class _DummyPlt:
    def get_cmap(self, name):
        return _DummyCMap()

    def isinteractive(self):
        return False

    def show(self, *_, **__):
        pass

    def pause(self, *_, **__):
        pass


@pytest.fixture
def patched_visualizer(monkeypatch):
    def fake_init(
        self,
        RIN_path: str | Path,
        *_,
        **__,
    ):
        with sawnergy_util.ArrayStorage(RIN_path, mode="r") as storage:
            com_name = storage.get_attr("com_name")
            attr_name = storage.get_attr("attractive_energies_name")
            rep_name = storage.get_attr("repulsive_energies_name")
            self.COM_coords = storage.read(com_name, slice(None))
            self.attr_energies = (
                storage.read(attr_name, slice(None)) if attr_name is not None else None
            )
            self.repuls_energies = (
                storage.read(rep_name, slice(None)) if rep_name is not None else None
            )
        self._fig = _DummyFigure()
        self._ax = _DummyAxes()
        self._scatter = _DummyScatter()
        self._attr = _DummyLineCollection()
        self._repuls = _DummyLineCollection()
        self.N = self.COM_coords.shape[1]
        self._residue_norm = _DummyNormalize()
        self.default_node_color = "#cccccc"
        self._plt = _DummyPlt()

    monkeypatch.setattr(visualizer_module.Visualizer, "__init__", fake_init)
    return None


@pytest.fixture
def rin_archive_path_both(patched_cpptraj: Path) -> Path:
    """RIN archive including both attractive and repulsive channels (energies+transitions)."""
    output_path = patched_cpptraj / "synthetic_rin_both.zip"
    builder = rin_builder.RINBuilder(cpptraj_path="cpptraj")
    builder.build_rin(
        topology_file="top.prmtop",
        trajectory_file="traj.nc",
        molecule_of_interest=1,
        frame_range=(1, FRAME_COUNT),
        frame_batch_size=1,
        prune_low_energies_frac=1.0,
        output_path=output_path,
        keep_prenormalized_energies=True,
        include_attractive=True,
        include_repulsive=True,
        compression_level=0,
        num_matrices_in_compressed_blocks=1,
    )
    return output_path


@pytest.fixture
def walks_archive_path_full(
    rin_archive_path_both: Path,
    tmp_path: Path,
    patched_shared_ndarray,
) -> Path:
    """Walks archive with both channels + SAWs present."""
    w = walker.Walker(rin_archive_path_both, seed=321)
    out_path = tmp_path / "synthetic_walks_full.zip"
    try:
        w.sample_walks(
            walk_length=8,
            walks_per_node=6,
            saw_frac=0.5,                       # both RW and SAW
            include_attractive=True,
            include_repulsive=True,             # repulsive channel path
            time_aware=False,
            output_path=out_path,
            in_parallel=False,
            compression_level=0,
            num_walk_matrices_in_compressed_blocks=1,
        )
    finally:
        w.close()
    return out_path


@pytest.fixture
def timeaware_walks_archive_path(
    rin_archive_path_both: Path,
    tmp_path: Path,
    patched_shared_ndarray,
) -> Path:
    """Walks archive sampled in time-aware mode (stickiness set)."""
    w = walker.Walker(rin_archive_path_both, seed=777)
    out_path = tmp_path / "synthetic_walks_timeaware.zip"
    try:
        w.sample_walks(
            walk_length=6,
            walks_per_node=3,
            saw_frac=0.0,
            include_attractive=True,
            include_repulsive=False,
            time_aware=True,                    # exercise _step_time path
            stickiness=0.75,
            on_no_options="loop",
            output_path=out_path,
            in_parallel=False,
            compression_level=0,
            num_walk_matrices_in_compressed_blocks=1,
        )
    finally:
        w.close()
    return out_path
