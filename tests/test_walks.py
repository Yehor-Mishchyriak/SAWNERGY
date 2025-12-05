from __future__ import annotations

import numpy as np
import pytest

from sawnergy import sawnergy_util
from sawnergy.walks import walker

from .conftest import (
    FRAME_COUNT,
    RESIDUE_COUNT,
    PAIRWISE_MATRICES,
    compute_processed_channels,
)


def test_walks_preserve_order(walks_archive_path):
    with sawnergy_util.ArrayStorage(walks_archive_path, mode="r") as storage:
        attr_name = storage.get_attr("attractive_RWs_name")
        walks = storage.read(attr_name, slice(None))
        assert storage.get_attr("repulsive_RWs_name") is None
        assert storage.get_attr("repulsive_SAWs_name") is None
        assert storage.get_attr("walk_length") == 2
        assert storage.get_attr("time_stamp_count") == FRAME_COUNT

    assert walks.shape[0] == FRAME_COUNT
    assert walks.shape[1] == RESIDUE_COUNT

    expected_starts = np.arange(1, RESIDUE_COUNT + 1, dtype=np.uint16)

    for frame_idx in range(FRAME_COUNT):
        starts = walks[frame_idx, :, 0]
        np.testing.assert_array_equal(starts, expected_starts)

    # ensure the per-frame walks differ, demonstrating frame ordering is preserved
    assert not np.array_equal(walks[0], walks[1])


def test_transition_vectors_match_rin(rin_archive_path, patched_shared_ndarray):
    w = walker.Walker(rin_archive_path, seed=123)
    try:
        for frame_idx, frame_id in enumerate(sorted(PAIRWISE_MATRICES)):
            _, _, expected = compute_processed_channels(PAIRWISE_MATRICES[frame_id])
            for node in range(RESIDUE_COUNT):
                prob = w._extract_prob_vector(node, frame_idx, "attr")
        np.testing.assert_allclose(prob, expected[node])
    finally:
        w.close()


def test_step_node_normalizes_probabilities(monkeypatch):
    stub = walker.Walker.__new__(walker.Walker)
    stub.nodes = np.array([0, 1, 2], dtype=np.intp)

    class _StubRNG:
        def __init__(self):
            self.last_p = None

        def choice(self, nodes, p):
            self.last_p = np.asarray(p, dtype=float)
            return int(nodes[0])

    stub.rng = _StubRNG()
    stub._extract_prob_vector = lambda *_: np.array([0.2, 0.3, 0.4], dtype=float) * 1.5  # sums to 1.35

    next_node, avoid = stub._step_node(node=0, interaction_type="attr", time_stamp=0, avoid=None)
    assert 0 <= next_node < stub.nodes.size
    assert avoid is None
    assert pytest.approx(1.0) == float(stub.rng.last_p.sum())
