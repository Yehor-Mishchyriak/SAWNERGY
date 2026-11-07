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
        saws = storage.read(storage.get_attr("attractive_SAWs_name"))
        assert saws.shape == (FRAME_COUNT, 0, 3) and saws.dtype == np.uint16
        assert storage.get_attr("num_RWs") == 1
        assert storage.get_attr("num_SAWs") == 0
        assert storage.get_attr("walks_layout") == "time_leading_3d"

    assert walks.shape[0] == FRAME_COUNT
    assert walks.shape[1] == RESIDUE_COUNT
    assert walks.shape[2] == 3 and walks.dtype == np.uint16
    assert np.all((walks >= 1) & (walks <= RESIDUE_COUNT))

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


@pytest.mark.parametrize("fixture_name", ["walks_archive_path_full", "timeaware_walks_archive_path"])
def test_walk_archive_channels_counts_and_start_order(request, fixture_name):
    path = request.getfixturevalue(fixture_name)
    with sawnergy_util.ArrayStorage(path, "r") as storage:
        length = storage.get_attr("walk_length")
        assert storage.get_attr("walks_layout") == "time_leading_3d"
        assert storage.get_attr("node_count") == RESIDUE_COUNT
        for channel in ("attractive", "repulsive"):
            for kind in ("RW", "SAW"):
                name = storage.get_attr(f"{channel}_{kind}s_name")
                if fixture_name == "timeaware_walks_archive_path" and channel == "repulsive":
                    assert name is None
                    continue
                assert name == f"{channel.upper()}_{kind}s"
                count = storage.get_attr(f"num_{kind}s")
                walks = storage.read(name)
                assert walks.dtype == np.uint16
                assert walks.shape == (FRAME_COUNT, RESIDUE_COUNT * count, length + 1)
                assert np.all((walks >= 1) & (walks <= RESIDUE_COUNT))
                expected_starts = np.repeat(np.arange(1, RESIDUE_COUNT + 1), count)
                for frame in walks:
                    np.testing.assert_array_equal(frame[:, 0], expected_starts)
