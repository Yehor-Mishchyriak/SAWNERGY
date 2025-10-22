from __future__ import annotations

from sawnergy import sawnergy_util

from .conftest import FRAME_COUNT, _StubSGNS


def test_embeddings_preserve_order(embeddings_archive_path):
    with sawnergy_util.ArrayStorage(embeddings_archive_path, mode="r") as storage:
        name = storage.get_attr("frame_embeddings_name")
        embeddings = storage.read(name, slice(None))

    assert embeddings.shape[0] == FRAME_COUNT
    assert len(_StubSGNS.call_log) == FRAME_COUNT
    assert _StubSGNS.call_log[0] != _StubSGNS.call_log[1]
    assert not (embeddings[0] == embeddings[1]).all()
