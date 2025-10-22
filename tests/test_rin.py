from __future__ import annotations

import numpy as np

from sawnergy import sawnergy_util
from sawnergy.rin import rin_builder

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


def test_rin_archive_preserves_frame_order(rin_archive_path):
    with sawnergy_util.ArrayStorage(rin_archive_path, mode="r") as storage:
        attrs = dict(storage.root.attrs)
        assert attrs["prune_low_energies_frac"] == 1.0
        assert attrs["molecule_of_interest"] == 1
        assert tuple(attrs["frame_range"]) == (1, len(PAIRWISE_MATRICES))

        attr_name = attrs["attractive_transitions_name"]
        rep_name = attrs["repulsive_transitions_name"]
        transitions = storage.read(attr_name, slice(None))
        assert transitions.shape[0] == len(PAIRWISE_MATRICES)
        assert rep_name is None

        energies_name = attrs["attractive_energies_name"]
        energies = storage.read(energies_name, slice(None))
        assert energies.shape == transitions.shape

        com_name = attrs["com_name"]
        com = storage.read(com_name, slice(None))
        assert com.shape == (len(PAIRWISE_MATRICES), RESIDUE_COUNT, 3)

    for idx, frame_id in enumerate(sorted(PAIRWISE_MATRICES)):
        frame_matrix = PAIRWISE_MATRICES[frame_id]
        attr_energy, _, attr_transition = compute_processed_channels(frame_matrix)
        np.testing.assert_allclose(energies[idx], attr_energy)
        np.testing.assert_allclose(transitions[idx], attr_transition)
        np.testing.assert_allclose(com[idx], COM_COORDS[frame_id])
