from __future__ import annotations

import numpy as np

from sawnergy import sawnergy_util

from .conftest import FRAME_COUNT, RESIDUE_COUNT


def test_walks_preserve_order(walks_archive_path):
    with sawnergy_util.ArrayStorage(walks_archive_path, mode="r") as storage:
        attr_name = storage.get_attr("attractive_RWs_name")
        walks = storage.read(attr_name, slice(None))

    if walks.ndim == 4:
        walks = walks[0]

    assert walks.shape[0] == FRAME_COUNT
    assert walks.shape[1] == RESIDUE_COUNT

    expected_starts = np.arange(1, RESIDUE_COUNT + 1, dtype=np.uint16)

    for frame_idx in range(FRAME_COUNT):
        starts = walks[frame_idx, :, 0]
        np.testing.assert_array_equal(starts, expected_starts)

    # ensure the per-frame walks differ, demonstrating frame ordering is preserved
    assert not np.array_equal(walks[0], walks[1])
