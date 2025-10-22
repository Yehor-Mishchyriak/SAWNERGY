from __future__ import annotations

import numpy as np

from sawnergy import sawnergy_util
from sawnergy.rin import rin_builder

from .conftest import COM_COORDS, PAIRWISE_MATRICES, RESIDUE_COUNT


def _expected_transition(matrix: np.ndarray) -> np.ndarray:
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
    return normalized_attr


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


def test_rin_archive_preserves_frame_order(rin_archive_path):
    with sawnergy_util.ArrayStorage(rin_archive_path, mode="r") as storage:
        attr_name = storage.get_attr("attractive_transitions_name")
        rep_name = storage.get_attr("repulsive_transitions_name")
        transitions = storage.read(attr_name, slice(None))
        assert transitions.shape[0] == 2
        assert rep_name is None

    expected_first = _expected_transition(PAIRWISE_MATRICES[1])
    expected_second = _expected_transition(PAIRWISE_MATRICES[2])

    np.testing.assert_allclose(transitions[0], expected_first)
    np.testing.assert_allclose(transitions[1], expected_second)
