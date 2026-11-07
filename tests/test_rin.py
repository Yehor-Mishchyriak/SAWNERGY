from __future__ import annotations

from dataclasses import FrozenInstanceError
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sawnergy import sawnergy_util
from sawnergy.rin import rin_builder, rin_util

from .conftest import (
    COM_COORDS,
    PAIRWISE_MATRICES,
    RESIDUE_COUNT,
    FRAME_COUNT,
    compute_processed_channels,
)


def test_cpptraj_regex_parsing(patched_cpptraj):
    builder = rin_builder.RINBuilder(cpptraj_path="cpptraj")
    matrix = builder._calc_avg_atomic_interactions_in_frames(
        (1, 1), "top.prmtop", "traj.nc", molecule_id=1
    )
    assert matrix.shape == (RESIDUE_COUNT, RESIDUE_COUNT)
    np.testing.assert_allclose(matrix, PAIRWISE_MATRICES[1])

    com_frames = builder._get_residue_COMs_per_frame(
        (1, 1), "top.prmtop", "traj.nc", molecule_id=1, number_residues=RESIDUE_COUNT
    )
    assert len(com_frames) == 1
    np.testing.assert_allclose(com_frames[0], COM_COORDS[1])


def test_rin_archive_attractive_channel(rin_archive_path):
    with sawnergy_util.ArrayStorage(rin_archive_path, mode="r") as storage:
        attrs = dict(storage.root.attrs)
        attr_name = attrs["attractive_transitions_name"]
        transitions = storage.read(attr_name, slice(None))
        energies = storage.read(attrs["attractive_energies_name"], slice(None))
        com = storage.read(attrs["com_name"], slice(None))

    assert transitions.shape[0] == len(PAIRWISE_MATRICES)
    assert energies.shape == transitions.shape
    assert com.shape == (len(PAIRWISE_MATRICES), RESIDUE_COUNT, 3)

    for idx, frame_id in enumerate(sorted(PAIRWISE_MATRICES)):
        attr_energy, _, attr_transition = compute_processed_channels(PAIRWISE_MATRICES[frame_id])
        np.testing.assert_allclose(energies[idx], attr_energy)
        np.testing.assert_allclose(transitions[idx], attr_transition)
        np.testing.assert_allclose(com[idx], COM_COORDS[frame_id])


def test_locate_cpptraj_deduplicates_candidates(monkeypatch, tmp_path):
    exe = tmp_path / "cpptraj"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)

    monkeypatch.setenv("CPPTRAJ", str(exe))

    def fake_which(name):
        if name.startswith("cpptraj"):
            return str(exe)
        return None

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setattr(os, "access", lambda path, mode: True)

    calls = []

    def fake_run(cmd, capture_output, text, timeout):
        calls.append(Path(cmd[0]))
        class _Res:
            returncode = 0
            stderr = ""
        return _Res()

    monkeypatch.setattr(subprocess, "run", fake_run)

    resolved = rin_util.locate_cpptraj(explicit=exe, verify=True)
    assert Path(resolved) == exe.resolve()
    assert len(calls) == 1  # explicit/env/PATH duplicates only probed once


def test_pairwise_nonfinite_values_raise_clear_error(monkeypatch):
    monkeypatch.setattr(rin_util, "locate_cpptraj", lambda explicit=None, verify=True: "/usr/bin/cpptraj")

    def fake_run_cpptraj(executable, argv=None, script=None, env=None, timeout=None):
        if argv and "-tl" in argv:
            return "Frames: 100"
        if script and "printdata PW[EMAP]" in script:
            return (
                "[printdata PW[EMAP] square2d noheader]\n"
                "0 inf\n"
                "-inf 0\n"
                "[printdata PW[VMAP] square2d noheader]\n"
                "0 1\n"
                "2 0\n"
            )
        return ""

    monkeypatch.setattr(rin_util, "run_cpptraj", fake_run_cpptraj)

    builder = rin_builder.RINBuilder(cpptraj_path="cpptraj")
    with pytest.raises(ValueError, match=r"Non-finite values .*frames 10\.\.10"):
        builder._calc_avg_atomic_interactions_in_frames(
            (10, 10), "top.prmtop", "traj.nc", molecule_id=1
        )


@pytest.mark.parametrize("script,expected", [
    (None, None),
    ("", "quit\n"),
    ("   ", "   \nquit\n"),
    ("run", "run\nquit\n"),
    ("run\n", "run\nquit\n"),
    ("run\r\n", "run\r\nquit\n"),
    ("run\n\n", "run\n\nquit\n"),
    ("writedata results.quit", "writedata results.quit\nquit\n"),
    ("printdata quit", "printdata quit\nquit\n"),
    ("help quit", "help quit\nquit\n"),
    ("run\n# quit", "run\n# quit\nquit\n"),
    ("run\n# quit\n", "run\n# quit\nquit\n"),
    ("quit", "quit"),
    ("run\nquit\n", "run\nquit\n"),
    ("run\n  QuIt  \n\n", "run\n  QuIt  \n\n"),
    ("run\r\nquit\r\n", "run\r\nquit\r\n"),
    ("run\nquit # finished", "run\nquit # finished"),
    ("run\nquit\n# trailing comment\n", "run\nquit\n# trailing comment\n"),
    ("run\nquit # finished\n\n# another comment\n",
     "run\nquit # finished\n\n# another comment\n"),
])
def test_run_cpptraj_script_termination_and_forwarding(monkeypatch, script, expected):
    executable = "/path with spaces/cpptraj"
    argv = ["-p", "topology with spaces.prmtop"]
    env = {"OMP_NUM_THREADS": "1"}
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="original\nstdout\n", stderr="warning")

    monkeypatch.setattr(subprocess, "run", fake_run)
    output = rin_util.run_cpptraj(executable, script=script, argv=argv, timeout=5, env=env)

    assert output == "original\nstdout\n"
    assert calls == [([executable] + argv, dict(input=expected, text=True,
                                              capture_output=True, check=True,
                                              timeout=5, env=env))]
    assert argv == ["-p", "topology with spaces.prmtop"]
    assert env == {"OMP_NUM_THREADS": "1"}


def test_run_cpptraj_default_arguments(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert rin_util.run_cpptraj("cpptraj") == ""
    assert calls == [(["cpptraj"], dict(input=None, text=True, capture_output=True,
                                       check=True, timeout=None, env=None))]


@pytest.mark.parametrize("error", [
    subprocess.CalledProcessError(2, ["cpptraj"], output="partial", stderr="failed"),
    subprocess.TimeoutExpired(["cpptraj"], 5, output="partial", stderr="waiting"),
    OSError("launch failed"),
    KeyboardInterrupt("interrupted"),
])
def test_run_cpptraj_preserves_subprocess_errors(monkeypatch, error):
    def fail_run(args, **kwargs):
        assert kwargs["input"] == "run\nquit\n"
        raise error

    monkeypatch.setattr(subprocess, "run", fail_run)
    with pytest.raises(type(error)) as caught:
        rin_util.run_cpptraj("cpptraj", script="run", timeout=5)
    assert caught.value is error


def test_cpptraj_script_composition_is_immutable():
    script = rin_util.CpptrajScript.from_cmd("parm top.prmtop")
    command = rin_util.CpptrajScript.from_cmd("trajin traj.nc")
    combined = script + command + "pairwise PW"
    redirected = combined > "matrix.dat"
    assert script.commands == ("parm top.prmtop",)
    assert command.commands == ("trajin traj.nc",)
    assert combined.commands == ("parm top.prmtop", "trajin traj.nc", "pairwise PW")
    assert redirected.commands == combined.commands[:-1] + ("pairwise PW out matrix.dat",)
    assert (script > command).commands == (script + command).commands
    assert script.__add__(1) is NotImplemented
    with pytest.raises(TypeError):
        script + 1
    with pytest.raises(FrozenInstanceError):
        script.commands = ()
    with pytest.raises(IndexError):
        rin_util.CpptrajScript() > "empty.dat"


@pytest.mark.parametrize("commands,expected", [
    ((), "run\n"), (("parm top",), "parm top\nrun\n"),
    (("run", "printdata PW"), "run\nprintdata PW\n"),
    (("RUN", " run "), "RUN\n run \nrun\n"),
])
def test_cpptraj_script_render_preserves_exact_run_convention(commands, expected):
    script = rin_util.CpptrajScript(commands)
    assert script.render() == expected
    assert script.commands == commands


def test_mask_parser_preserves_ids_order_and_collapses_duplicate_rows(tmp_path):
    path = tmp_path / "composition.txt"
    path.write_text(
        "[Name] [Rnum] [AtNum] [Mnum]\n"
        "C 7 20 3\n\nN\t2\t9\t1\nO 7 4 3\nC 7 20 3\nH 8 3 3\n"
    )
    hierarchy = rin_util.CpptrajMaskParser.hierarchize_molecular_composition(str(path))
    assert hierarchy == {3: {7: {20, 4}, 8: {3}}, 1: {2: {9}}}
    assert list(hierarchy) == [3, 1]
    assert list(hierarchy[3]) == [7, 8]


@pytest.mark.parametrize("text,expected", [
    ("[Mnum] [Rnum] [AtNum]\n\n", {}),
    ("[Mnum] [Rnum] [AtNum]\n0 -2 -4\n", {0: {-2: {-4}}}),
    ("[Mnum] [Rnum] [AtNum] [Mnum]\n9 2 3 1\n", {1: {2: {3}}}),
])
def test_mask_parser_preserves_legacy_header_and_id_behavior(tmp_path, text, expected):
    path = tmp_path / "composition.txt"
    path.write_text(text)
    assert rin_util.CpptrajMaskParser.hierarchize_molecular_composition(str(path)) == expected


@pytest.mark.parametrize("text,error", [
    ("", RuntimeError), ("[Mnum] [Rnum]\n", ValueError),
    ("[Mnum] [Rnum] [AtNum]\n1 2\n", ValueError),
    ("[Mnum] [Rnum] [AtNum]\n1 two 3\n", ValueError),
])
def test_mask_parser_rejects_malformed_input(tmp_path, text, error):
    path = tmp_path / "composition.txt"
    path.write_text(text)
    with pytest.raises(error):
        rin_util.CpptrajMaskParser.hierarchize_molecular_composition(str(path))


def test_mask_parser_propagates_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        rin_util.CpptrajMaskParser.hierarchize_molecular_composition(str(tmp_path / "missing"))


@pytest.fixture
def cpptraj_candidates(tmp_path, monkeypatch):
    paths = [tmp_path / name / "bin" / "cpptraj" for name in
             ("explicit", "environment", "path", "path_exe", "amber", "conda")]
    for path in paths:
        path.parent.mkdir(parents=True)
        path.touch()
        path.chmod(0o755)
    monkeypatch.setenv("CPPTRAJ", str(paths[1]))
    monkeypatch.setattr(shutil, "which", lambda name: str(paths[2 if name == "cpptraj" else 3]))
    monkeypatch.setenv("AMBERHOME", str(paths[4].parent.parent))
    monkeypatch.setenv("CONDA_PREFIX", str(paths[5].parent.parent))
    return paths


def test_locate_cpptraj_search_order_timeout_and_return_code_one(cpptraj_candidates, monkeypatch):
    paths = cpptraj_candidates
    calls = []

    def probe(args, **kwargs):
        calls.append(args)
        assert kwargs == dict(capture_output=True, text=True, timeout=15)
        if args[0] == str(paths[0]):
            raise subprocess.TimeoutExpired(args, 15)
        return subprocess.CompletedProcess(args, 1 if args[0] == str(paths[-1]) else 2, stderr="unavailable")

    monkeypatch.setattr(subprocess, "run", probe)
    assert rin_util.locate_cpptraj(paths[0]) == str(paths[-1].resolve())
    assert calls == [[str(path), "-h"] for path in paths]


def test_locate_cpptraj_skips_missing_and_nonexecutable_without_probe(cpptraj_candidates, monkeypatch):
    paths = cpptraj_candidates
    paths[0].unlink()
    monkeypatch.setattr(os, "access", lambda path, mode: path != paths[1])

    def unexpected_probe(*args, **kwargs):
        pytest.fail("verify=False must not launch CPPTRAJ")

    monkeypatch.setattr(subprocess, "run", unexpected_probe)
    assert rin_util.locate_cpptraj(paths[0], verify=False) == str(paths[2].resolve())


def test_locate_cpptraj_exhaustion_lists_candidates(cpptraj_candidates, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs:
                        subprocess.CompletedProcess(args, 2, stderr="unavailable"))
    with pytest.raises(rin_util.CpptrajNotFound) as caught:
        rin_util.locate_cpptraj(cpptraj_candidates[0])
    assert isinstance(caught.value, RuntimeError)
    positions = [str(caught.value).index(str(path)) for path in cpptraj_candidates]
    assert positions == sorted(positions)


def test_locate_cpptraj_propagates_launch_error(cpptraj_candidates, monkeypatch):
    error = OSError("cannot launch")

    def fail_probe(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", fail_probe)
    with pytest.raises(OSError) as caught:
        rin_util.locate_cpptraj(cpptraj_candidates[0])
    assert caught.value is error


def test_rin_archive_preserves_both_channels_and_metadata(rin_archive_path_both):
    with sawnergy_util.ArrayStorage(rin_archive_path_both, "r") as storage:
        assert storage.get_attr("molecule_of_interest") == 1
        assert storage.get_attr("frame_range") == [1, FRAME_COUNT]
        assert storage.get_attr("frame_batch_size") == 1
        assert storage.get_attr("prune_low_energies_frac") == 1.0
        expected_blocks = {"COM"}
        for channel, energy_index in [("attractive", 0), ("repulsive", 1)]:
            energy_name = storage.get_attr(f"{channel}_energies_name")
            transition_name = storage.get_attr(f"{channel}_transitions_name")
            assert energy_name == f"{channel.upper()}_energies"
            assert transition_name == f"{channel.upper()}_transitions"
            expected_blocks.update([energy_name, transition_name])
            energy = storage.read(energy_name)
            transition = storage.read(transition_name)
            assert energy.shape == transition.shape == (FRAME_COUNT, RESIDUE_COUNT, RESIDUE_COUNT)
            assert energy.dtype == transition.dtype == np.float32
            for index, frame in enumerate(sorted(PAIRWISE_MATRICES)):
                expected_energy = compute_processed_channels(PAIRWISE_MATRICES[frame])[energy_index]
                sums = expected_energy.sum(axis=1, keepdims=True)
                expected_transition = expected_energy / np.maximum(sums, 1e-12)
                np.testing.assert_allclose(energy[index], expected_energy)
                np.testing.assert_allclose(transition[index], expected_transition)
                np.testing.assert_allclose(energy[index], energy[index].T)
                assert np.all(energy[index] >= 0)
                assert np.all(np.diag(energy[index]) == 0)
        assert set(storage.list_blocks()) == expected_blocks


@pytest.mark.parametrize("frame_range,batch_size,windows", [
    (None, -1, [(1, 2)]), ((1, 2), 0, [(1, 2)]), ((2, 2), 1, [(2, 2)]),
    ((1, 2), 1, [(1, 1), (2, 2)]),
])
def test_rin_frame_selection_com_means_and_disabled_energy_blocks(
    patched_cpptraj, frame_range, batch_size, windows,
):
    output = patched_cpptraj / "selected.zip"
    builder = rin_builder.RINBuilder(cpptraj_path="cpptraj")
    result = builder.build_rin(
        "top.prmtop", "traj.nc", 1, frame_range=frame_range, frame_batch_size=batch_size,
        output_path=output, keep_prenormalized_energies=False, include_repulsive=False,
        compression_level=0, num_matrices_in_compressed_blocks=1,
    )
    assert result == str(output)
    with sawnergy_util.ArrayStorage(result, "r") as storage:
        assert storage.get_attr("frame_range") == (list(frame_range) if frame_range else None)
        assert storage.get_attr("frame_batch_size") == (batch_size if batch_size > 0 else FRAME_COUNT)
        assert storage.get_attr("attractive_energies_name") is None
        assert storage.get_attr("repulsive_energies_name") is None
        assert storage.get_attr("repulsive_transitions_name") is None
        assert storage.list_blocks() == ["ATTRACTIVE_transitions", "COM"]
        expected = np.stack([
            np.mean([COM_COORDS[frame] for frame in range(start, end + 1)], axis=0)
            for start, end in windows
        ])
        actual = storage.read(storage.get_attr("com_name"))
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, expected)
        assert storage.read("ATTRACTIVE_transitions").shape[0] == len(windows)


def test_rin_normalization_preserves_zero_rows_and_denominator_floor():
    builder = rin_builder.RINBuilder.__new__(rin_builder.RINBuilder)
    channels = np.array([
        [[0, 0], [1e-13, 0]], [[0, 2], [2, 0]],
    ], dtype=np.float32)
    normalized = builder._L1_normalize(channels)
    np.testing.assert_allclose(normalized, [
        [[0, 0], [0.1, 0]], [[0, 1], [1, 0]],
    ])


def test_rin_partial_final_batch_preserves_com_means(patched_cpptraj, monkeypatch):
    monkeypatch.setitem(COM_COORDS, 3, np.full((RESIDUE_COUNT, 3), 9, dtype=np.float32))
    monkeypatch.setitem(PAIRWISE_MATRICES, 3, PAIRWISE_MATRICES[1])
    run = rin_util.run_cpptraj

    def run_three_frames(executable, argv=None, **kwargs):
        return "Frames: 3" if argv and "-tl" in argv else run(executable, argv=argv, **kwargs)

    monkeypatch.setattr(rin_util, "run_cpptraj", run_three_frames)
    output = patched_cpptraj / "partial_batch.zip"
    builder = rin_builder.RINBuilder(cpptraj_path="cpptraj")
    builder.build_rin(
        "top.prmtop", "traj.nc", 1, frame_range=(1, 3), frame_batch_size=2,
        output_path=output, compression_level=0, num_matrices_in_compressed_blocks=1,
    )
    with sawnergy_util.ArrayStorage(output, "r") as storage:
        expected = np.stack([(COM_COORDS[1] + COM_COORDS[2]) / 2, COM_COORDS[3]])
        np.testing.assert_allclose(storage.read("COM"), expected)
        assert storage.read("ATTRACTIVE_transitions").shape[0] == 2
        assert storage.read("REPULSIVE_transitions").shape[0] == 2


def test_residue_aggregation_sums_atom_pairs_in_sorted_id_order():
    builder = rin_builder.RINBuilder
    membership = builder._compute_residue_membership_matrix({8: {30}, 2: {20, 10}})
    np.testing.assert_array_equal(membership, [[1, 0], [1, 0], [0, 1]])
    atomic = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    residue = builder._convert_atomic_to_residue_interactions(atomic, membership)
    # Sum the four blocks explicitly: atoms (10,20) belong to residue 2; 30 to 8.
    np.testing.assert_array_equal(residue, [[12, 9], [15, 9]])
    assert residue.dtype == np.float32
