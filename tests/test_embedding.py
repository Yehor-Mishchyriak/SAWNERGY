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
        assert embeddings.dtype == np.float32
        assert storage.get_attr("time_stamp_count") == FRAME_COUNT
        assert storage.get_attr("model_base") == "torch"
        assert storage.get_attr("node_count") == embeddings.shape[1]
        assert storage.get_attr("embedding_dim") == embeddings.shape[2]
        assert storage.get_attr("embedding_kind") == "in"
        assert storage.get_attr("objective") == "sgns"
        assert storage.get_attr("negative_sampling") is True
        assert storage.get_attr("num_epochs") == 1
        assert storage.get_attr("num_negative_samples") == 1
        assert storage.get_attr("batch_size") == 4
        assert storage.get_attr("window_size") == 1
        assert storage.get_attr("alpha") == pytest.approx(0.75)
        assert storage.get_attr("RIN_type") == "attr"
        assert storage.get_attr("using") == "RW"
        assert storage.get_attr("master_seed") == 999

    assert embeddings.shape[0] == FRAME_COUNT
    assert len(_StubSGNS.call_log) == FRAME_COUNT

    master = np.random.SeedSequence(999)
    expected_seeds = [int(seq.generate_state(1, dtype=np.uint32)[0]) for seq in master.spawn(FRAME_COUNT)]
    assert _StubSGNS.call_log == expected_seeds

    for idx, seed in enumerate(expected_seeds):
        rng = np.random.default_rng(seed)
        base = rng.random()
        values = np.linspace(0.0, 1.0, embeddings.shape[2], dtype=np.float32)
        expected_row = values + base
        expected_emb = np.tile(expected_row, (embeddings.shape[1], 1))
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


def test_as_zerobase_intp_bounds_and_dtype():
    W = np.array([[1, 2, 3], [3, 2, 1]], dtype=np.uint16)  # 1-based
    out = embedder_module.Embedder._as_zerobase_intp(W, V=4)
    assert out.dtype == np.intp and out.min() == 0 and out.max() == 2
    with pytest.raises(ValueError):
        embedder_module.Embedder._as_zerobase_intp(np.array([[0, 1]]), V=2)  # 0 not allowed after 1→0
    with pytest.raises(ValueError):
        embedder_module.Embedder._as_zerobase_intp(np.array([[2, 5]]), V=4)  # 4 out of range after shift


def test_soft_unigram_properties():
    f = np.array([0, 2, 6, 2], dtype=int)
    p1 = embedder_module.Embedder._soft_unigram(f, power=1.0)
    np.testing.assert_allclose(p1, np.array([0.0, 0.2, 0.6, 0.2]))
    with pytest.raises(ValueError):
        embedder_module.Embedder._soft_unigram(np.zeros_like(f))


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

    # Prefer avg_embeddings when present; fall back to embeddings.
    embeddings = getattr(model, "avg_embeddings", getattr(model, "embeddings", None))
    assert embeddings is not None
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

    weights = getattr(model, "avg_embeddings", getattr(model, "embeddings", None))
    assert weights is not None
    assert np.isfinite(weights).all()


def test_sg_pureml_smoke(monkeypatch):
    """Plain SG (full softmax) with PureML — loss drops, embeddings finite."""
    pureml = pytest.importorskip("pureml")
    Tensor = pureml.machinery.Tensor
    CCE = pureml.losses.CCE
    one_hot = pureml.training_utils.one_hot
    optim_cls = getattr(pureml.optimizers, "Adam", None)
    if optim_cls is None:
        pytest.skip("pureml optim.Adam unavailable")

    from sawnergy.embedding.SGNS_pml import SG_PureML

    if getattr(SG_PureML, "__call__", None) is not SG_PureML.predict:
        monkeypatch.setattr(SG_PureML, "__call__", SG_PureML.predict)

    model = SG_PureML(
        V=5, D=4, seed=123,
        optim=optim_cls, optim_kwargs=dict(lr=1e-2),
        lr_sched=None, lr_sched_kwargs=None,
        device=None,
    )
    rng = np.random.default_rng(0)
    centers  = rng.integers(0, 5, size=20, dtype=np.int64)
    contexts = rng.integers(0, 5, size=20, dtype=np.int64)

    def _loss(m):
        logits = m(Tensor(centers))
        y = one_hot(5, label=Tensor(contexts))
        return float(CCE(y, logits, from_logits=True).numpy())

    before = _loss(model)
    model.fit(
        centers, contexts,
        num_epochs=3, batch_size=5,
        shuffle_data=False, lr_step_per_batch=False,
    )
    after = _loss(model)
    assert after <= before
    W = getattr(model, "avg_embeddings", getattr(model, "embeddings", None))
    assert W is not None
    assert np.isfinite(W).all()


def test_sg_torch_smoke():
    """Plain SG (full softmax) with Torch — loss drops, embeddings finite."""
    torch = pytest.importorskip("torch")
    from sawnergy.embedding.SGNS_torch import SG_Torch
    optim_cls = getattr(torch.optim, "Adam", None)
    if optim_cls is None:
        pytest.skip("torch optim.Adam unavailable")

    model = SG_Torch(
        V=5, D=4, seed=123,
        optim=optim_cls, optim_kwargs=dict(lr=1e-2),
        lr_sched=None, lr_sched_kwargs=None,
        device=None,
    )
    rng = np.random.default_rng(0)
    centers  = torch.as_tensor(rng.integers(0, 5, size=20), dtype=torch.long)
    contexts = torch.as_tensor(rng.integers(0, 5, size=20), dtype=torch.long)

    cce = torch.nn.CrossEntropyLoss(reduction="mean")
    def _loss(m):
        logits = m(centers)
        return float(cce(logits, contexts).item())

    before = _loss(model)
    model.fit(
        centers.numpy(), contexts.numpy(),
        num_epochs=3, batch_size=5,
        shuffle_data=False, lr_step_per_batch=False,
    )
    after = _loss(model)
    assert after <= before
    W = getattr(model, "avg_embeddings", getattr(model, "embeddings", None))
    assert W is not None
    assert np.isfinite(W).all()


# ---------------------------------------------------------------------------
# New tests: embed_frame ordering + warm-start forwarding (SGNS vs SG)
# ---------------------------------------------------------------------------

def _expected_stub_embeds(V: int, D: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Helper to compute the deterministic embeddings produced by _StubSGNS.fit()."""
    rng = np.random.default_rng(seed)
    base = float(rng.random())
    ramp = np.linspace(0.0, 1.0, D, dtype=np.float32)
    in_row = base + ramp
    out_row = base + 0.5 + ramp
    E_in = np.tile(in_row, (V, 1)).astype(np.float32)
    E_out = np.tile(out_row, (V, 1)).astype(np.float32)
    return E_in, E_out


def test_embed_frame_returns_sorted_kinds_and_numpy(walks_archive_path, patched_sgns):
    _StubSGNS.call_log.clear()
    _StubSGNS.init_log.clear()

    emb = embedder_module.Embedder(walks_archive_path, seed=111)
    # Ask in weird order -> Embedder sorts to ('avg','in','out')
    out = emb.embed_frame(
        frame_id=1, RIN_type="attr", using="RW",
        num_epochs=1, negative_sampling=True,
        window_size=1, num_negative_samples=1, batch_size=4,
        kind=("out", "avg", "in"),
    )
    kinds = [k for (_, k) in out]
    assert kinds == ["avg", "in", "out"]
    for arr, _ in out:
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32


def test_warm_starts_forwarding_sgns(walks_archive_path, patched_sgns):
    """SGNS: out warm start should be (V, D)."""
    _StubSGNS.call_log.clear()
    _StubSGNS.init_log.clear()

    seed = 2024
    D = 3
    emb = embedder_module.Embedder(walks_archive_path, seed=seed)
    V = emb.vocab_size

    # Run across 2 frames (dataset fixture provides FRAME_COUNT=2)
    emb.embed_all(
        RIN_type="attr", using="RW",
        num_epochs=1, negative_sampling=True,
        window_size=1, num_negative_samples=1, batch_size=8,
        dimensionality=D, model_base="torch", model_kwargs={},
    )

    # Two model constructions, one per frame
    assert len(_StubSGNS.init_log) == FRAME_COUNT

    # Frame 1: no warm starts
    first = _StubSGNS.init_log[0]
    assert first["in_weights_shape"] is None
    assert first["out_weights_shape"] is None

    # Frame 2: warm starts from frame 1
    second = _StubSGNS.init_log[1]
    assert second["in_weights_shape"] == (V, D)
    assert second["out_weights_shape"] == (V, D)

    # Compute expected frame-1 embeddings from seeds the Embedder uses
    master = np.random.SeedSequence(seed)
    child_seeds = [int(s.generate_state(1, dtype=np.uint32)[0]) for s in master.spawn(FRAME_COUNT)]
    first_seed = child_seeds[0]
    E_in_expected, E_out_expected = _expected_stub_embeds(V, D, first_seed)

    np.testing.assert_allclose(second["in_weights"], E_in_expected, rtol=0, atol=0)
    np.testing.assert_allclose(second["out_weights"], E_out_expected, rtol=0, atol=0)


def test_warm_starts_forwarding_sg_transposed_out(walks_archive_path, patched_sgns):
    """SG (full softmax): out warm start should be (D, V) i.e., transpose of previous 'out' (V, D)."""
    _StubSGNS.call_log.clear()
    _StubSGNS.init_log.clear()

    seed = 3031
    D = 4
    emb = embedder_module.Embedder(walks_archive_path, seed=seed)
    V = emb.vocab_size

    emb.embed_all(
        RIN_type="attr", using="RW",
        num_epochs=1, negative_sampling=False,  # SG path
        window_size=1, num_negative_samples=1, batch_size=8,
        dimensionality=D, model_base="torch", model_kwargs={},
    )

    assert len(_StubSGNS.init_log) == FRAME_COUNT

    first = _StubSGNS.init_log[0]
    assert first["in_weights_shape"] is None
    assert first["out_weights_shape"] is None

    second = _StubSGNS.init_log[1]
    assert second["in_weights_shape"] == (V, D)
    # Key difference vs SGNS:
    assert second["out_weights_shape"] == (D, V)

    # Expected transpose check
    master = np.random.SeedSequence(seed)
    child_seeds = [int(s.generate_state(1, dtype=np.uint32)[0]) for s in master.spawn(FRAME_COUNT)]
    first_seed = child_seeds[0]
    _, E_out_expected = _expected_stub_embeds(V, D, first_seed)
    np.testing.assert_allclose(second["out_weights"], E_out_expected.T, rtol=0, atol=0)
