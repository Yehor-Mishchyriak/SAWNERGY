from __future__ import annotations

import numpy as np

import pytest

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

    master = np.random.SeedSequence(999)
    expected_seeds = [int(seq.generate_state(1, dtype=np.uint32)[0]) for seq in master.spawn(FRAME_COUNT)]
    assert _StubSGNS.call_log == expected_seeds

    for idx, seed in enumerate(expected_seeds):
        rng = np.random.default_rng(seed)
        base = rng.random()
        values = np.linspace(0.0, 1.0, embeddings.shape[2], dtype=np.float32)
        expected_emb = (values + base).repeat(embeddings.shape[1]).reshape(embeddings.shape[1], embeddings.shape[2])
        np.testing.assert_allclose(embeddings[idx], expected_emb)


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


def test_pairs_from_walks_randomized():
    rng = np.random.default_rng(0)
    for window_size in [1, 2, 3]:
        for _ in range(20):
            num_walks = rng.integers(1, 4)
            walk_len = rng.integers(0, 5)
            vocab = rng.integers(1, 6)
            walks = rng.integers(0, vocab, size=(num_walks, walk_len), dtype=np.intp)
            pairs = embedder_module.Embedder._pairs_from_walks(walks, window_size)
            expected = set()
            for row in walks:
                L = row.shape[0]
                for i in range(L):
                    for d in range(1, window_size + 1):
                        if i + d < L:
                            expected.add((row[i], row[i + d]))
                        if i - d >= 0:
                            expected.add((row[i], row[i - d]))
            assert set(map(tuple, pairs.tolist())) == expected

    empty_pairs = embedder_module.Embedder._pairs_from_walks(np.zeros((1, 0), dtype=np.intp), window_size=2)
    assert empty_pairs.size == 0

    single_pairs = embedder_module.Embedder._pairs_from_walks(np.array([[0]], dtype=np.intp), window_size=2)
    assert single_pairs.size == 0


def test_sgns_pureml_smoke(monkeypatch):
    pureml = pytest.importorskip("pureml")
    Tensor = pureml.machinery.Tensor
    BCE = pureml.losses.BCE
    optim_cls = getattr(pureml.optimizers, "Adam", None)
    if optim_cls is None:
        pytest.skip("pureml optim.Adam unavailable")

    class _Scheduler:
        def __init__(self, **kwargs):
            pass

        def step(self):
            return None

    from sawnergy.embedding.SGNS_pml import SGNS_PureML

    if getattr(SGNS_PureML, "__call__", None) is not SGNS_PureML.predict:
        monkeypatch.setattr(SGNS_PureML, "__call__", SGNS_PureML.predict)

    model = SGNS_PureML(
        V=4,
        D=3,
        seed=123,
        optim=optim_cls,
        optim_kwargs={"lr": 0.05},
        lr_sched=_Scheduler,
        lr_sched_kwargs={},
    )

    centers = np.array([0, 1, 2, 3], dtype=np.int64)
    contexts = np.array([1, 2, 3, 0], dtype=np.int64)
    negatives = np.array([[2, 3], [3, 0], [0, 1], [1, 2]], dtype=np.int64)
    noise = np.full(model.V, 1 / model.V, dtype=np.float64)

    def _loss(model_obj):
        pos_logits, neg_logits = model_obj.predict(
            Tensor(centers), Tensor(contexts), Tensor(negatives)
        )
        y_pos = Tensor(np.ones_like(pos_logits.data))
        y_neg = Tensor(np.zeros_like(neg_logits.data))
        loss = BCE(y_pos, pos_logits, from_logits=True) + BCE(y_neg, neg_logits, from_logits=True)
        return float(loss.data.mean())

    before = _loss(model)
    for _ in range(3):
        pos_logits, neg_logits = model.predict(
            Tensor(centers), Tensor(contexts), Tensor(negatives)
        )
        y_pos = Tensor(np.ones_like(pos_logits.data))
        y_neg = Tensor(np.zeros_like(neg_logits.data))
        loss = BCE(y_pos, pos_logits, from_logits=True) + BCE(y_neg, neg_logits, from_logits=True)
        model.optim.zero_grad()
        loss.backward()
        model.optim.step()
        model.lr_sched.step()
    after = _loss(model)
    assert after <= before
    embeddings = model.embeddings
    assert np.isfinite(embeddings).all()


def test_sgns_torch_smoke():
    torch = pytest.importorskip("torch")
    from sawnergy.embedding.SGNS_torch import SGNS_Torch

    model = SGNS_Torch(
        V=4,
        D=3,
        seed=123,
        optim=torch.optim.Adam,
        optim_kwargs={"lr": 0.05},
        lr_sched=None,
        lr_sched_kwargs=None,
        device="cpu",
    )

    centers = np.array([0, 1, 2, 3], dtype=np.int64)
    contexts = np.array([1, 2, 3, 0], dtype=np.int64)
    negatives = np.array([[2, 3], [3, 0], [0, 1], [1, 2]], dtype=np.int64)
    noise = np.full(model.V, 1 / model.V, dtype=np.float64)

    bce = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def _loss(model_obj):
        pos_logits, neg_logits = model_obj.predict(
            torch.as_tensor(centers, dtype=torch.long),
            torch.as_tensor(contexts, dtype=torch.long),
            torch.as_tensor(negatives, dtype=torch.long),
        )
        y_pos = torch.ones_like(pos_logits)
        y_neg = torch.zeros_like(neg_logits)
        loss = bce(pos_logits, y_pos) + bce(neg_logits, y_neg)
        return float(loss.item())

    before = _loss(model)
    model.fit(
        centers,
        contexts,
        num_epochs=3,
        batch_size=2,
        num_negative_samples=2,
        noise_dist=noise,
        shuffle_data=False,
        lr_step_per_batch=False,
    )
    after = _loss(model)
    assert after <= before
    weights = model.embeddings
    assert np.isfinite(weights).all()
