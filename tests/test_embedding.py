from __future__ import annotations

import numpy as np

from sawnergy import sawnergy_util
from sawnergy.embedding import embedder as embedder_module

from .conftest import FRAME_COUNT, _StubSGNS


def test_embeddings_preserve_order(embeddings_archive_path):
    with sawnergy_util.ArrayStorage(embeddings_archive_path, mode="r") as storage:
        name = storage.get_attr("frame_embeddings_name")
        embeddings = storage.read(name, slice(None))
        assert storage.get_attr("frames_written") == FRAME_COUNT
        assert storage.get_attr("frame_count") == FRAME_COUNT
        assert storage.get_attr("model_base") == "torch"
        assert storage.get_attr("num_negative_samples") == 1

    assert embeddings.shape[0] == FRAME_COUNT
    assert len(_StubSGNS.call_log) == FRAME_COUNT
    assert _StubSGNS.call_log[0] != _StubSGNS.call_log[1]
    assert not np.allclose(embeddings[0], embeddings[1])


def test_pairs_from_walks_skipgram_window_one():
    walks = np.array([[0, 1, 2, 3]], dtype=np.intp)
    pairs = embedder_module.Embedder._pairs_from_walks(walks, window_size=1)
    expected = {
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
    }
    assert len(pairs) == len(expected)
    assert set(map(tuple, pairs.tolist())) == expected
