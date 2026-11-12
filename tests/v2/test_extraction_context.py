from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sawnergy.sawnergy_util import ArrayStorage
from sawnergy.v2.traj_features.feature_extractor import DependencyReader, ExtractionContext


@pytest.fixture
def context_inputs(tmp_path):
    return dict(cpptraj=tmp_path / "cpptraj", topology_file=tmp_path / "topology.prmtop",
                trajectory_file=tmp_path / "trajectory.nc",
                residue_ids=[(1, 1), (2, 3)], atom_ids=[[1, 2], [5]],
                frame_ids=[2, 4, 6], windows=[[2, 4], [6]])


def test_context_snapshots_inputs_and_freezes_fields(context_inputs):
    context_inputs["cpptraj"] = str(context_inputs["cpptraj"])
    dependencies = {}
    ctx = ExtractionContext(**context_inputs, dependencies=dependencies)
    context_inputs["residue_ids"][0] = (1, 2)
    context_inputs["atom_ids"][0].append(3)
    context_inputs["frame_ids"].append(8)
    context_inputs["windows"][0].append(6)
    dependencies["later"] = object()

    assert isinstance(ctx.cpptraj, Path) and ctx.cpptraj.is_absolute()
    assert ctx.topology_file == context_inputs["topology_file"].resolve()
    assert ctx.trajectory_file == context_inputs["trajectory_file"].resolve()
    assert ctx.selection_mask == ":1,3"
    assert ctx.residue_ids == ((1, 1), (2, 3))
    assert ctx.atom_ids == ((1, 2), (5,))
    assert ctx.frame_ids == (2, 4, 6)
    assert ctx.windows == ((2, 4), (6,))
    assert dict(ctx.dependencies) == {}
    with pytest.raises(FrozenInstanceError):
        ctx.selection_mask = ":2"
    with pytest.raises(TypeError):
        ctx.dependencies["later"] = object()


def test_context_allows_explicit_frames_without_windows(context_inputs):
    del context_inputs["windows"]
    ctx = ExtractionContext(**context_inputs)
    assert ctx.frame_ids == (2, 4, 6)
    assert ctx.windows == ()


def test_context_selection_mask_is_derived_from_fixed_residue_ids(context_inputs):
    ctx = ExtractionContext(**context_inputs)
    assert ctx.selection_mask == ":1,3"
    with pytest.raises(TypeError):
        ExtractionContext(**context_inputs, selection_mask=":1<:5.0")


def test_context_allows_overlapping_windows(context_inputs):
    context_inputs["windows"] = [[2, 4], [4, 6]]
    assert ExtractionContext(**context_inputs).windows == ((2, 4), (4, 6))


@pytest.mark.parametrize("field,value", [
    ("residue_ids", []), ("residue_ids", [(1,)]), ("residue_ids", [(True, 1)]),
    ("residue_ids", [(0, 1)]), ("residue_ids", [(1, True)]),
    ("residue_ids", [(1, 3), (2, 1)]), ("residue_ids", [(1, 1), (2, 1)]),
    ("atom_ids", [[1, 2]]), ("atom_ids", [[1, 2], []]),
    ("atom_ids", [[2, 1], [5]]), ("atom_ids", [[1, 2], [2]]),
    ("atom_ids", [[1, 1], [5]]), ("atom_ids", [[0, 2], [5]]),
    ("atom_ids", [[True, 2], [5]]),
    ("frame_ids", []), ("frame_ids", [2, 2, 6]), ("frame_ids", [4, 2, 6]),
    ("frame_ids", [0, 4, 6]), ("frame_ids", [True, 4, 6]),
    ("frame_ids", [2.0, 4, 6]),
    ("windows", [[]]), ("windows", [[2, 2], [4, 6]]),
    ("windows", [[4, 2], [6]]), ("windows", [[True, 4], [6]]),
    ("windows", [[2, 4], [6, 8]]), ("windows", [[2], [6]]),
    ("dependencies", {"": object()}), ("dependencies", {"energy": object()}),
])
def test_context_rejects_invalid_values(context_inputs, field, value):
    context_inputs[field] = value
    with pytest.raises(ValueError):
        ExtractionContext(**context_inputs)


@pytest.mark.parametrize("field", ["cpptraj", "topology_file", "trajectory_file",
                                   "residue_ids", "atom_ids", "frame_ids", "windows"])
def test_context_rejects_invalid_path_and_sequence_types(context_inputs, field):
    context_inputs[field] = None
    with pytest.raises(TypeError):
        ExtractionContext(**context_inputs)


@pytest.mark.parametrize("compressed", [False, True])
def test_dependency_reader_slices_batches_and_copies(tmp_path, context_inputs, compressed):
    values = np.arange(20, dtype=np.float64).reshape(5, 2, 2)
    with ArrayStorage(tmp_path / "features", "w") as writer:
        writer.write(values, "energy", arrays_per_chunk=2)
        writer.write(values + 100, "other", arrays_per_chunk=2)
        path = writer.compress(tmp_path / "features.zip", compression_level=0) if compressed else writer.store_path

    with ArrayStorage(path, "r") as storage:
        reader = DependencyReader(storage, "energy")
        original_dependencies = {"electrostatics": reader}
        ctx = ExtractionContext(**context_inputs, dependencies=original_dependencies)
        original_dependencies.clear()
        assert ctx.dependencies["electrostatics"] is reader
        np.testing.assert_array_equal(reader.read(2), values[2])
        np.testing.assert_array_equal(reader.read(slice(1, 5, 2)), values[1:5:2])
        np.testing.assert_array_equal(reader.read((4, 0)), values[[4, 0]])
        batches = list(reader.block_iter(step=2))
        assert [len(batch) for batch in batches] == [2, 2, 1]
        np.testing.assert_array_equal(np.concatenate(batches), values)
        read = reader.read()
        read[:] = -1
        batches[0][:] = -2
        np.testing.assert_array_equal(reader.read(), values)


def test_dependency_reader_defers_reads_and_preserves_storage_errors(tmp_path, monkeypatch):
    with ArrayStorage(tmp_path / "features", "w") as storage:
        error = OSError("read failed")

        def failed_read(*args, **kwargs):
            raise error

        monkeypatch.setattr(storage, "read", failed_read)
        reader = DependencyReader(storage, "energy")
        with pytest.raises(OSError) as caught:
            reader.read(slice(0, 1))
        assert caught.value is error


@pytest.mark.parametrize("block_name", ["", None, 1])
def test_dependency_reader_rejects_invalid_block_name(tmp_path, block_name):
    with ArrayStorage(tmp_path / "features", "w") as storage:
        with pytest.raises(ValueError):
            DependencyReader(storage, block_name)


def test_dependency_reader_rejects_invalid_storage():
    with pytest.raises(TypeError):
        DependencyReader(object(), "energy")


def test_context_validation_remains_active_under_optimization():
    source = """
from sawnergy.v2.traj_features.feature_extractor import ExtractionContext
inputs = dict(cpptraj='cpptraj', topology_file='topology', trajectory_file='trajectory',
              residue_ids=((1, 1),), atom_ids=((1,),),
              frame_ids=(1,), windows=((1,),))
ctx = ExtractionContext(**inputs)
for field, value in (('frame_ids', (True,)), ('atom_ids', ((),)),
                     ('residue_ids', ((0, 1),)), ('windows', ((2,),)),
                     ('dependencies', {'energy': object()})):
    try:
        ExtractionContext(**dict(inputs, **{field: value}))
    except ValueError:
        continue
    raise RuntimeError(f'Invalid {field} was accepted')
try:
    ctx.dependencies['energy'] = object()
except TypeError:
    pass
else:
    raise RuntimeError('Dependency mapping was mutable')
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    result = subprocess.run([sys.executable, "-O", "-c", source], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
