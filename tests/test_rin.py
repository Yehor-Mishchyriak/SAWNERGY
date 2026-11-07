from __future__ import annotations

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
