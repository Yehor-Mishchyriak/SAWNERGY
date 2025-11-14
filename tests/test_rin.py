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
