import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from sawnergy.rin import rin_util
from sawnergy.v2.traj_features.feature_extractor import FeatureExtractor


@pytest.fixture
def backend(tmp_path, monkeypatch):
    topology, trajectory, cpptraj = (tmp_path / name for name in ("top with spaces", "traj with spaces", "cpptraj"))
    for path in (topology, trajectory, cpptraj):
        path.write_text("fixture")
    cpptraj.chmod(0o755)
    state = SimpleNamespace(inputs=dict(topology_file=topology, trajectory_file=trajectory,
                                       cpptraj_path=cpptraj, selection={2: [5], 1: [2, 1]}),
                            calls=[], tables=[], trajectory_reads=[], count="Frames: 11\n", error=None)
    monkeypatch.setattr(rin_util, "locate_cpptraj", lambda explicit, verify=True: str(explicit))

    def run(cpptraj, script=None, argv=None):
        state.calls.append((cpptraj, script, argv))
        if script is None:
            return state.count
        table = Path(re.search(r'mask :\* out "([^"]+)"', script)[1])
        trajectory = re.search(r'trajin "([^"]+)" (\d+) (\d+)', script)
        state.trajectory_reads.append((Path(trajectory[1]).resolve(), int(trajectory[2]), int(trajectory[3])))
        state.tables.append(table)
        if state.error:
            raise state.error
        table.write_text("[AtNum] [Rnum] [Mnum]\n9 5 2\n4 2 1\n2 1 1\n8 5 2\n3 2 1\n1 1 1\n")
        return "Read 1 frames and processed 1 frames.\n"

    monkeypatch.setattr(rin_util, "run_cpptraj", run)
    return state


def test_preparation_resolves_complete_fixed_selection_in_topology_order(backend):
    ctx = FeatureExtractor()._prepare_inputs(**backend.inputs, frame_range=(2, 11),
                                           frame_stride=3, window_size=3)
    assert ctx.residue_ids == ((1, 1), (1, 2), (2, 5))
    assert ctx.atom_ids == ((1, 2), (3, 4), (8, 9))
    assert ctx.selection_mask == ":1,2,5"
    assert ctx.frame_ids == (2, 5, 8, 11)
    assert ctx.windows == ((2, 5, 8), (11,))
    assert dict(ctx.dependencies) == {}
    assert backend.inputs["selection"] == {2: [5], 1: [2, 1]}
    assert backend.trajectory_reads == [(ctx.trajectory_file, 2, 2)]
    assert all(not table.exists() for table in backend.tables)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 20])
def test_window_batches_preserve_selected_frames_and_partial_windows(backend, batch_size):
    extractor = FeatureExtractor()
    ctx = extractor._prepare_inputs(**backend.inputs, frame_stride=2, window_size=2)
    batches = list(extractor._window_batches(ctx, batch_size=batch_size))
    assert ctx.frame_ids == (1, 3, 5, 7, 9, 11)
    assert ctx.windows == ((1, 3), (5, 7), (9, 11))
    assert tuple(window for batch in batches for window in batch.windows) == ctx.windows
    assert tuple(frame for batch in batches for frame in batch.frame_ids) == ctx.frame_ids
    assert all(len(batch.windows) <= batch_size for batch in batches)
    assert all(batch.residue_ids == ctx.residue_ids and batch.atom_ids == ctx.atom_ids for batch in batches)
    assert all(batch.selection_mask == ctx.selection_mask for batch in batches)
    assert all(batch.topology_file == ctx.topology_file and batch.trajectory_file == ctx.trajectory_file
               and batch.cpptraj == ctx.cpptraj for batch in batches)


@pytest.mark.parametrize("frame_range,stride,size,frames,windows", [
    ((4, 4), 3, 5, (4,), ((4,),)),
    ((2, 10), 3, 2, (2, 5, 8), ((2, 5), (8,))),
    (None, 20, 1, (1,), ((1,),)),
])
def test_frame_selection_boundaries(backend, frame_range, stride, size, frames, windows):
    ctx = FeatureExtractor()._prepare_inputs(**backend.inputs, frame_range=frame_range,
                                           frame_stride=stride, window_size=size)
    assert ctx.frame_ids == frames
    assert ctx.windows == windows


@pytest.mark.parametrize("selection,error", [
    (":1,2", TypeError), (":1<:5.0", TypeError), (None, TypeError),
    ({}, ValueError), ({1: []}, ValueError), ({1: [1, 1]}, ValueError),
    ({True: [1]}, ValueError), ({0: [1]}, ValueError), ({1: [True]}, ValueError),
    ({1: [0]}, ValueError), ({1: [-1]}, ValueError), ({1: [1.0]}, ValueError),
    ({1: ["1"]}, ValueError), ({1: None}, TypeError),
])
def test_preparation_rejects_invalid_selection_before_cpptraj(backend, selection, error):
    backend.inputs["selection"] = selection
    with pytest.raises(error):
        FeatureExtractor()._prepare_inputs(**backend.inputs)
    assert backend.calls == []


@pytest.mark.parametrize("selection", [{9: [1]}, {1: [5]}, {2: [1]}, {1: [1, 99]}])
def test_preparation_rejects_every_unknown_or_wrong_molecule_residue(backend, selection):
    backend.inputs["selection"] = selection
    with pytest.raises(ValueError, match="residue"):
        FeatureExtractor()._prepare_inputs(**backend.inputs)
    assert all(not table.exists() for table in backend.tables)


@pytest.mark.parametrize("arguments", [
    {"frame_range": (0, 2)}, {"frame_range": (2, 1)}, {"frame_range": (True, 2)},
    {"frame_range": (1,)}, {"frame_range": [1, 2]},
    {"frame_stride": 0}, {"frame_stride": True}, {"frame_stride": 1.5},
    {"window_size": 0}, {"window_size": -1}, {"window_size": True},
])
def test_preparation_rejects_invalid_frame_parameters_before_cpptraj(backend, arguments):
    with pytest.raises(ValueError):
        FeatureExtractor()._prepare_inputs(**backend.inputs, **arguments)
    assert backend.calls == []


@pytest.mark.parametrize("frame_range", [(1, 12), (12, 12)])
def test_preparation_rejects_out_of_bounds_frames(backend, frame_range):
    with pytest.raises(ValueError):
        FeatureExtractor()._prepare_inputs(**backend.inputs, frame_range=frame_range)
    assert len(backend.calls) == 1


@pytest.mark.parametrize("count,error", [("Frames: 0", ValueError), ("Frames: -1", ValueError),
                                         ("Frames: unknown", RuntimeError)])
def test_preparation_rejects_empty_or_malformed_trajectory_counts(backend, count, error):
    backend.count = count
    with pytest.raises(error):
        FeatureExtractor()._prepare_inputs(**backend.inputs)


@pytest.mark.parametrize("name", ["topology_file", "trajectory_file", "cpptraj_path"])
def test_preparation_requires_source_files(backend, name):
    backend.inputs[name].unlink()
    with pytest.raises(FileNotFoundError):
        FeatureExtractor()._prepare_inputs(**backend.inputs)
    assert backend.calls == []


def test_preparation_requires_executable_cpptraj(backend):
    backend.inputs["cpptraj_path"].chmod(0o644)
    with pytest.raises(PermissionError):
        FeatureExtractor()._prepare_inputs(**backend.inputs)
    assert backend.calls == []


def test_preparation_rejects_fallback_from_explicit_cpptraj(backend, monkeypatch):
    monkeypatch.setattr(rin_util, "locate_cpptraj", lambda explicit, verify=True: "/different/cpptraj")
    with pytest.raises(rin_util.CpptrajNotFound):
        FeatureExtractor()._prepare_inputs(**backend.inputs)
    assert backend.calls == []


def test_preparation_preserves_backend_failure_and_cleans_temporary_files(backend):
    backend.error = subprocess.CalledProcessError(1, "cpptraj")
    with pytest.raises(subprocess.CalledProcessError) as caught:
        FeatureExtractor()._prepare_inputs(**backend.inputs)
    assert caught.value is backend.error
    assert backend.tables and all(not table.exists() for table in backend.tables)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_window_batches_reject_invalid_sizes(backend, batch_size):
    extractor = FeatureExtractor()
    ctx = extractor._prepare_inputs(**backend.inputs)
    with pytest.raises(ValueError):
        list(extractor._window_batches(ctx, batch_size=batch_size))


def test_input_validation_under_optimization(backend):
    inputs = {key: str(value) if isinstance(value, Path) else value for key, value in backend.inputs.items()}
    source = f"inputs = {inputs!r}\n" + """
from sawnergy.v2.traj_features.feature_extractor import ExtractionContext, FeatureExtractor
extractor = FeatureExtractor()
for changes in ({'selection': ':1<:5.0'}, {'selection': {}}, {'selection': {True: [1]}},
                {'frame_range': (0, 2)}, {'frame_stride': True}, {'window_size': 0}):
    try:
        extractor._prepare_inputs(**dict(inputs, **changes))
    except (TypeError, ValueError):
        continue
    raise RuntimeError('Invalid preparation inputs were accepted')
ctx = ExtractionContext(cpptraj=inputs['cpptraj_path'], topology_file=inputs['topology_file'],
                        trajectory_file=inputs['trajectory_file'], residue_ids=((1, 1),),
                        atom_ids=((1,),), frame_ids=(1,), windows=((1,),))
if ctx.selection_mask != ':1':
    raise RuntimeError('Selection mask was not derived from the fixed IDs')
for size in (0, True, 1.5):
    try:
        list(extractor._window_batches(ctx, batch_size=size))
    except ValueError:
        continue
    raise RuntimeError('Invalid batch size was accepted')
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    result = subprocess.run([sys.executable, "-O", "-c", source], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_input_preparation_uses_actual_topology_and_frame_selection(tmp_path, monkeypatch):
    cpptraj = os.environ.get("CPPTRAJ") or shutil.which("cpptraj")
    if not cpptraj:
        pytest.skip("Set CPPTRAJ to run the native input-preparation regression")
    example = Path(__file__).resolve().parents[2] / "example_MD_for_quick_start"
    topology, trajectory = tmp_path / "topology with spaces.prmtop", tmp_path / "trajectory with spaces.nc"
    shutil.copyfile(example / "p53_DBD.prmtop", topology)
    shutil.copyfile(example / "p53_DBD.nc", trajectory)
    run = rin_util.run_cpptraj

    def checked_run(*args, **kwargs):
        output = run(*args, **kwargs)
        if kwargs.get("script"):
            assert "Read 1 frames and processed 1 frames." in output
        return output

    monkeypatch.setattr(rin_util, "run_cpptraj", checked_run)
    ctx = FeatureExtractor()._prepare_inputs(topology, trajectory, cpptraj, {1: [3, 1]}, frame_range=(2, 11),
                                           frame_stride=3, window_size=3)
    assert ctx.residue_ids == ((1, 1), (1, 3))
    assert ctx.atom_ids == (tuple(range(1, 14)), tuple(range(30, 44)))
    assert ctx.selection_mask == ":1,3"
    assert ctx.frame_ids == (2, 5, 8, 11)
    assert ctx.windows == ((2, 5, 8), (11,))
