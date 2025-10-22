from __future__ import annotations

"""
Embedding orchestration for Skip-Gram with Negative Sampling (SGNS).

This module consumes attractive/repulsive walk corpora produced by the walker
pipeline and trains per-frame embeddings using either the PyTorch or PureML
implementations of SGNS. The resulting embeddings can be persisted back into
an ``ArrayStorage`` archive along with rich metadata describing the training
configuration.
"""

# third-pary
import numpy as np

# built-in
from pathlib import Path
from typing import Literal
import logging

# local
from .. import sawnergy_util

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__)

# *----------------------------------------------------*
#                        CLASSES
# *----------------------------------------------------*

class Embedder:
    """Skip-gram embedder over attractive/repulsive walk corpora."""

    def __init__(self,
                 WALKS_path: str | Path,
                 base: Literal["torch", "pureml"],
                 *,
                 seed: int | None = None
                ) -> None:
        """Initialize the embedder and load walk tensors.

        Args:
            WALKS_path: Path to a ``WALKS_*.zip`` (or ``.zarr``) archive created
                by the walker pipeline. The archive's root attrs must include:
                ``attractive_RWs_name``, ``repulsive_RWs_name``,
                ``attractive_SAWs_name``, ``repulsive_SAWs_name`` (each may be
                ``None`` if that collection is absent), and the metadata
                ``num_RWs``, ``num_SAWs``, ``node_count``, ``time_stamp_count``,
                ``walk_length``.
            base: Which SGNS backend to use, either ``"torch"`` or ``"pureml"``.
            seed: Optional seed for the embedder's RNG. If ``None``, a random
                32-bit seed is chosen.

        Raises:
            ValueError: If required metadata is missing or any loaded walk array
                has an unexpected shape.
            ImportError: If the requested backend is not installed.
            NameError: If ``base`` is not one of ``{"torch","pureml"}``.

        Notes:
            - Walks in storage are 1-based (residue indexing). Internally, this
              class normalizes to 0-based indices for training utilities.
        """
        self._walks_path = Path(WALKS_path)
        _logger.info("Initializing Embedder from %s (base=%s)", self._walks_path, base)

        # placeholders for optional walk collections
        self.attractive_RWs : np.ndarray | None = None
        self.repulsive_RWs  : np.ndarray | None = None
        self.attractive_SAWs: np.ndarray | None = None
        self.repulsive_SAWs : np.ndarray | None = None

        # Load numpy arrays from read-only storage
        with sawnergy_util.ArrayStorage(self._walks_path, mode="r") as storage:
            attractive_RWs_name   = storage.get_attr("attractive_RWs_name")
            repulsive_RWs_name    = storage.get_attr("repulsive_RWs_name")
            attractive_SAWs_name  = storage.get_attr("attractive_SAWs_name")
            repulsive_SAWs_name   = storage.get_attr("repulsive_SAWs_name")

            attractive_RWs  : np.ndarray | None = (
                storage.read(attractive_RWs_name, slice(None)) if attractive_RWs_name is not None else None
            )

            repulsive_RWs  : np.ndarray | None = (
                storage.read(repulsive_RWs_name, slice(None)) if repulsive_RWs_name is not None else None
            )

            attractive_SAWs  : np.ndarray | None = (
                storage.read(attractive_SAWs_name, slice(None)) if attractive_SAWs_name is not None else None
            )

            repulsive_SAWs  : np.ndarray | None = (
                storage.read(repulsive_SAWs_name, slice(None)) if repulsive_SAWs_name is not None else None
            )

            num_RWs          = storage.get_attr("num_RWs")
            num_SAWs         = storage.get_attr("num_SAWs")
            node_count       = storage.get_attr("node_count")
            time_stamp_count = storage.get_attr("time_stamp_count")
            walk_length      = storage.get_attr("walk_length")

        if node_count is None or time_stamp_count is None or walk_length is None:
            raise ValueError("WALKS metadata missing one of node_count, time_stamp_count, walk_length")

        _logger.debug(
            ("Loaded WALKS from %s"
             " | ATTR RWs: %s %s"
             " | REP  RWs: %s %s"
             " | ATTR SAWs: %s %s"
             " | REP  SAWs: %s %s"
             " | num_RWs=%d num_SAWs=%d V=%d L=%d T=%d"),
            self._walks_path,
            getattr(attractive_RWs, "shape", None), getattr(attractive_RWs, "dtype", None),
            getattr(repulsive_RWs, "shape", None),  getattr(repulsive_RWs, "dtype", None),
            getattr(attractive_SAWs, "shape", None), getattr(attractive_SAWs, "dtype", None),
            getattr(repulsive_SAWs, "shape", None),  getattr(repulsive_SAWs, "dtype", None),
            num_RWs, num_SAWs, node_count, walk_length, time_stamp_count
        )

        # expected shapes
        RWs_expected  = (time_stamp_count, node_count * num_RWs,  walk_length+1) if (num_RWs  > 0) else None
        SAWs_expected = (time_stamp_count, node_count * num_SAWs, walk_length+1) if (num_SAWs > 0) else None

        self.vocab_size = int(node_count)
        self.frame_count = int(time_stamp_count)
        self.walk_length = int(walk_length)

        # store walks if present
        if attractive_RWs is not None:
            if RWs_expected and attractive_RWs.shape != RWs_expected:
                raise ValueError(f"ATTR RWs: expected {RWs_expected}, got {attractive_RWs.shape}")
            self.attractive_RWs = attractive_RWs

        if repulsive_RWs is not None:
            if RWs_expected and repulsive_RWs.shape != RWs_expected:
                raise ValueError(f"REP RWs: expected {RWs_expected}, got {repulsive_RWs.shape}")
            self.repulsive_RWs = repulsive_RWs

        if attractive_SAWs is not None:
            if SAWs_expected and attractive_SAWs.shape != SAWs_expected:
                raise ValueError(f"ATTR SAWs: expected {SAWs_expected}, got {attractive_SAWs.shape}")
            self.attractive_SAWs = attractive_SAWs

        if repulsive_SAWs is not None:
            if SAWs_expected and repulsive_SAWs.shape != SAWs_expected:
                raise ValueError(f"REP SAWs: expected {SAWs_expected}, got {repulsive_SAWs.shape}")
            self.repulsive_SAWs = repulsive_SAWs

        # INTERNAL RNG
        self._seed = np.random.randint(0, 2**32 - 1) if seed is None else int(seed)
        self.rng = np.random.default_rng(self._seed)
        _logger.info("RNG initialized from seed=%d", self._seed)

        # MODEL HANDLE
        self.model_base: Literal["torch", "pureml"] = base
        self.model_constructor = self._get_SGNS_constructor_from(base)
        _logger.info("SGNS backend resolved: %s", getattr(self.model_constructor, "__name__", repr(self.model_constructor)))

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- PRIVATE -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # HELPERS:

    @staticmethod
    def _get_SGNS_constructor_from(base: Literal["torch", "pureml"]):
        """Resolve the SGNS implementation class for the selected backend."""
        if base == "torch":
            try:
                from .SGNS_torch import SGNS_Torch
                return SGNS_Torch
            except Exception:
                raise ImportError(
                    "PyTorch is not installed, but based_on='torch' was requested. "
                    "Install PyTorch first, e.g.: `pip install torch` "
                    "(see https://pytorch.org/get-started for platform-specific wheels)."
                )
        elif base == "pureml":
            try:
                from .SGNS_pml import SGNS_PureML
                return SGNS_PureML
            except Exception:
                raise ImportError(
                    "PureML is not installed, but based_on='pureml' was requested. "
                    "Install PureML first via `pip install ym-pure-ml` "
                )
        else:
            raise NameError(f"Expected `base` in (\"torch\", \"pureml\"); Instead got: {base}")

    @staticmethod
    def _as_zerobase_intp(walks: np.ndarray, *, V: int) -> np.ndarray:
        """Validate 1-based uint/int walks → 0-based intp; check bounds."""
        arr = np.asarray(walks)
        if arr.ndim != 2:
            raise ValueError("walks must be 2D: (num_walks, walk_len)")
        if arr.dtype.kind not in "iu":
            arr = arr.astype(np.int64, copy=False)
        # 1-based → 0-based
        arr = arr - 1
        if arr.min() < 0 or arr.max() >= V:
            raise ValueError("walk ids out of range after 1→0-based normalization")
        return arr.astype(np.intp, copy=False)

    @staticmethod
    def _pairs_from_walks(walks0: np.ndarray, window_size: int) -> np.ndarray:
        """
        Skip-gram pairs including edge centers (one-sided when needed).
        walks0: (W, L) int array (0-based ids).
        Returns: (N_pairs, 2) int32 [center, context].
        """
        if walks0.ndim != 2:
            raise ValueError("walks must be 2D: (num_walks, walk_len)")

        _, L = walks0.shape
        k = int(window_size)

        if k <= 0:
            raise ValueError("window_size must be positive")
        
        if L == 0:
            return np.empty((0, 2), dtype=np.int32)

        out_chunks = []
        for d in range(1, k + 1):
            span = L - d
            if span <= 0:
                break
            # right contexts: center j pairs with j+d  (centers 0..L-d-1)
            centers_r = walks0[:, :L - d]
            ctx_r     = walks0[:, d:]
            out_chunks.append(np.stack((centers_r, ctx_r), axis=2).reshape(-1, 2))
            # left contexts: center j pairs with j-d   (centers d..L-1)
            centers_l = walks0[:, d:]
            ctx_l     = walks0[:, :L - d]
            out_chunks.append(np.stack((centers_l, ctx_l), axis=2).reshape(-1, 2))

        if not out_chunks:
            return np.empty((0, 2), dtype=np.int32)

        return np.concatenate(out_chunks, axis=0).astype(np.int32, copy=False)

    @staticmethod
    def _freq_from_walks(walks0: np.ndarray, *, V: int) -> np.ndarray:
        """Node frequencies from walks (0-based)."""
        return np.bincount(walks0.ravel(), minlength=V).astype(np.int64, copy=False)

    @staticmethod
    def _soft_unigram(freq: np.ndarray, *, power: float = 0.75) -> np.ndarray:
        """Return normalized Pn(w) ∝ f(w)^power as float64 probs."""
        p = np.asarray(freq, dtype=np.float64)
        if p.sum() == 0:
            raise ValueError("all frequencies are zero")
        p = np.power(p, float(power))
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            raise ValueError("invalid unigram mass")
        return p / s

    def _materialize_walks(self, frame_id: int, rin: Literal["attr", "repuls"],
                           using: Literal["RW", "SAW", "merged"]) -> np.ndarray:
        if not 1 <= frame_id <= int(self.frame_count):
            raise IndexError(f"frame_id must be in [1, {self.frame_count}]; got {frame_id}")

        frame_id -= 1

        if rin == "attr":
            parts = []
            if using in ("RW", "merged"):
                arr = getattr(self, "attractive_RWs", None)
                if arr is not None:
                    parts.append(arr[frame_id])
            if using in ("SAW", "merged"):
                arr = getattr(self, "attractive_SAWs", None)
                if arr is not None:
                    parts.append(arr[frame_id])
        else:
            parts = []
            if using in ("RW", "merged"):
                arr = getattr(self, "repulsive_RWs", None)
                if arr is not None:
                    parts.append(arr[frame_id])
            if using in ("SAW", "merged"):
                arr = getattr(self, "repulsive_SAWs", None)
                if arr is not None:
                    parts.append(arr[frame_id])

        if not parts:
            raise ValueError(f"No walks available for {rin=} with {using=}")
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=0)

    # INTERFACES: (private)

    def _attractive_corpus_and_prob(self, *,
                                    frame_id: int,
                                    using: Literal["RW", "SAW", "merged"],
                                    window_size: int,
                                    alpha: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
        walks = self._materialize_walks(frame_id, "attr", using)
        walks0 = self._as_zerobase_intp(walks, V=self.vocab_size)
        attractive_corpus = self._pairs_from_walks(walks0, window_size)
        attractive_noise_probs = self._soft_unigram(self._freq_from_walks(walks0, V=self.vocab_size), power=alpha)
        _logger.info("ATTR corpus ready: pairs=%d", 0 if attractive_corpus is None else attractive_corpus.shape[0])
        
        return attractive_corpus, attractive_noise_probs

    def _repulsive_corpus_and_prob(self, *,
                                   frame_id: int,
                                   using: Literal["RW", "SAW", "merged"],
                                   window_size: int,
                                   alpha: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
        walks = self._materialize_walks(frame_id, "repuls", using)
        walks0 = self._as_zerobase_intp(walks, V=self.vocab_size)
        repulsive_corpus = self._pairs_from_walks(walks0, window_size)
        repulsive_noise_probs = self._soft_unigram(self._freq_from_walks(walks0, V=self.vocab_size), power=alpha)
        _logger.info("REP corpus ready: pairs=%d", 0 if repulsive_corpus is None else repulsive_corpus.shape[0])

        return repulsive_corpus, repulsive_noise_probs

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= PUBLIC -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= 

    def embed_frame(self,
              frame_id: int,
              RIN_type: Literal["attr", "repuls"],
              using: Literal["RW", "SAW", "merged"],
              window_size: int,
              num_negative_samples: int,
              num_epochs: int,
              batch_size: int,
              *,
              shuffle_data: bool = True,
              dimensionality: int = 128,
              alpha: float = 0.75,
              device: str | None = None,
              sgns_kwargs: dict[str, object] | None = None,
              _seed: int | None = None
              ) -> np.ndarray:
        """Train embeddings for a single frame and return the input embedding matrix.

        Args:
            frame_id: 1-based frame index to train on.
            RIN_type: Interaction channel to use: ``"attr"`` (attractive) or
                ``"repuls"`` (repulsive).
            using: Which walk collections to include: ``"RW"``, ``"SAW"``, or
                ``"merged"`` (concatenates both if available).
            window_size: Symmetric skip-gram window size ``k``.
            num_negative_samples: Number of negative samples per positive pair.
            num_epochs: Number of passes over the pair dataset.
            batch_size: Mini-batch size for training.
            shuffle_data: Whether to shuffle pairs each epoch.
            dimensionality: Embedding dimensionality ``D``.
            alpha: Noise distribution exponent (``Pn ∝ f^alpha``).
            device: Optional device string for the Torch backend (e.g., ``"cuda"``).
            sgns_kwargs: Extra keyword arguments forwarded to the backend SGNS
                constructor. For PureML, required keys are:
                ``{"optim", "optim_kwargs", "lr_sched", "lr_sched_kwargs"}``.
            _seed: Optional child seed for this frame's model initialization.

        Returns:
            np.ndarray: Learned **input** embedding matrix of shape ``(V, D)``.

        Raises:
            ValueError: If requested walks are missing, if no training pairs are
                generated, or if required ``sgns_kwargs`` for PureML are absent.
            AttributeError: If the SGNS model does not expose embeddings via
                ``.embeddings`` or ``.parameters[0]``.
        """
        _logger.info(
            "Preparing frame %d (rin=%s using=%s window=%d neg=%d epochs=%d batch=%d)",
            frame_id, RIN_type, using, window_size, num_negative_samples, num_epochs, batch_size
        )

        if RIN_type == "attr":
            if self.attractive_RWs is None and self.attractive_SAWs is None:
                raise ValueError("Attractive random walks are missing")
            pairs, noise_probs = self._attractive_corpus_and_prob(frame_id=frame_id, using=using, window_size=window_size, alpha=alpha)
        elif RIN_type == "repuls":
            if self.repulsive_RWs is None and self.repulsive_SAWs is None:
                raise ValueError("Repulsive random walks are missing")
            pairs, noise_probs = self._repulsive_corpus_and_prob(frame_id=frame_id, using=using, window_size=window_size, alpha=alpha)
        else:
            raise ValueError(f"Unknown RIN_type: {RIN_type!r}")

        if pairs.size == 0:
            raise ValueError("No training pairs generated for the requested configuration")

        centers  = pairs[:, 0].astype(np.int64, copy=False)
        contexts = pairs[:, 1].astype(np.int64, copy=False)

        model_kwargs: dict[str, object] = dict(sgns_kwargs or {})
        if self.model_base == "pureml":
            required = {"optim", "optim_kwargs", "lr_sched", "lr_sched_kwargs"}
            missing = required.difference(model_kwargs)
            if missing:
                raise ValueError(f"PureML backend requires {sorted(missing)} in sgns_kwargs.")

        child_seed = int(self._seed if _seed is None else _seed)
        model_kwargs.update({
            "V": self.vocab_size,
            "D": dimensionality,
            "seed": child_seed
        })

        if self.model_base == "torch" and device is not None:
            model_kwargs["device"] = device

        self.model = self.model_constructor(**model_kwargs)

        _logger.info(
            "Training SGNS base=%s constructor=%s frame=%d pairs=%d dim=%d epochs=%d batch=%d neg=%d shuffle=%s",
            self.model_base,
            getattr(self.model_constructor, "__name__", repr(self.model_constructor)),
            frame_id,
            pairs.shape[0],
            dimensionality,
            num_epochs,
            batch_size,
            num_negative_samples,
            shuffle_data
        )

        self.model.fit(
            centers,
            contexts,
            num_epochs,
            batch_size,
            num_negative_samples,
            noise_probs,
            shuffle_data,
            lr_step_per_batch=False
        )

        embeddings = getattr(self.model, "embeddings", None)
        if embeddings is None:
            params = getattr(self.model, "parameters", None)
            if isinstance(params, tuple) and params:
                embeddings = params[0]
        if embeddings is None:
            raise AttributeError("SGNS model does not expose embeddings via '.embeddings' or '.parameters[0]'")

        embeddings = np.asarray(embeddings)
        _logger.info("Frame %d embeddings ready: shape=%s dtype=%s", frame_id, embeddings.shape, embeddings.dtype)
        return embeddings

    def embed_all(
        self,
        RIN_type: Literal["attr", "repuls"],
        using: Literal["RW", "SAW", "merged"],
        window_size: int,
        num_negative_samples: int,
        num_epochs: int,
        batch_size: int,
        *,
        shuffle_data: bool = True,
        dimensionality: int = 128,
        alpha: float = 0.75,
        device: str | None = None,
        sgns_kwargs: dict[str, object] | None = None,
        output_path: str | Path | None = None,
        num_matrices_in_compressed_blocks: int = 20,
        compression_level: int = 3):
        """Train embeddings for all frames and persist them to compressed storage.

        Iterates through all frames (``1..frame_count``), trains an SGNS model
        per frame using the configured backend, collects the resulting input
        embeddings, and writes them into a new compressed ``ArrayStorage`` archive.

        Args:
            RIN_type: Interaction channel to use: ``"attr"`` or ``"repuls"``.
            using: Walk collections: ``"RW"``, ``"SAW"``, or ``"merged"``.
            window_size: Symmetric skip-gram window size ``k``.
            num_negative_samples: Number of negative samples per positive pair.
            num_epochs: Number of epochs for each frame.
            batch_size: Mini-batch size used during training.
            shuffle_data: Whether to shuffle pairs each epoch.
            dimensionality: Embedding dimensionality ``D``.
            alpha: Noise distribution exponent (``Pn ∝ f^alpha``).
            device: Optional device string for Torch backend.
            sgns_kwargs: Extra constructor kwargs for the SGNS backend (see
                :meth:`embed_frame` for PureML requirements).
            output_path: Destination path. If ``None``, a new file named
                ``EMBEDDINGS_<timestamp>.zip`` is created next to the source
                WALKS archive. If the provided path lacks a suffix, ``.zip`` is
                appended.
            num_matrices_in_compressed_blocks: Number of per-frame matrices to
                store per compressed chunk in the output archive.
            compression_level: Blosc Zstd compression level (0-9).

        Returns:
            str: Filesystem path to the written embeddings archive (``.zip``).

        Raises:
            ValueError: If configuration produces no pairs for a frame or if
                PureML kwargs are incomplete.
            RuntimeError: Propagated from storage operations on failure.

        Notes:
            - A deterministic child seed is spawned per frame from the master
              seed using ``np.random.SeedSequence`` to ensure reproducibility
              across runs.
        """
        current_time = sawnergy_util.current_time()
        if output_path is None:
            output_path = self._walks_path.with_name(f"EMBEDDINGS_{current_time}").with_suffix(".zip")
        else:
            output_path = Path(output_path)
            if output_path.suffix == "":
                output_path = output_path.with_suffix(".zip")

        _logger.info(
            "Embedding all frames -> %s | frames=%d dim=%d base=%s",
            output_path, self.frame_count, dimensionality, self.model_base
        )

        master_ss = np.random.SeedSequence(self._seed)
        child_seeds = master_ss.spawn(self.frame_count)

        embeddings = []
        for frame_idx, seed_seq in enumerate(child_seeds, start=1):
            child_seed = int(seed_seq.generate_state(1, dtype=np.uint32)[0])
            _logger.info("Processing frame %d/%d (child_seed=%d entropy=%d)", frame_idx, self.frame_count, child_seed, seed_seq.entropy)
            embeddings.append(
                self.embed_frame(
                    frame_idx,
                    RIN_type,
                    using,
                    window_size,
                    num_negative_samples,
                    num_epochs,
                    batch_size,
                    shuffle_data=shuffle_data,
                    dimensionality=dimensionality,
                    alpha=alpha,
                    device=device,
                    sgns_kwargs=sgns_kwargs,
                    _seed=child_seed
                )
            )

        embeddings = [np.asarray(e) for e in embeddings]
        block_name = "FRAME_EMBEDDINGS"
        with sawnergy_util.ArrayStorage.compress_and_cleanup(output_path, compression_level=compression_level) as storage:
            storage.write(
                these_arrays=embeddings,
                to_block_named=block_name,
                arrays_per_chunk=num_matrices_in_compressed_blocks
            )
            storage.add_attr("time_created", current_time)
            storage.add_attr("seed", int(self._seed))
            storage.add_attr("rng_scheme", "SeedSequence.spawn_per_frame_v1")
            storage.add_attr("source_walks_path", str(self._walks_path))
            storage.add_attr("model_base", self.model_base)
            storage.add_attr("rin_type", RIN_type)
            storage.add_attr("using_mode", using)
            storage.add_attr("window_size", int(window_size))
            storage.add_attr("alpha", float(alpha))
            storage.add_attr("dimensionality", int(dimensionality))
            storage.add_attr("num_negative_samples", int(num_negative_samples))
            storage.add_attr("num_epochs", int(num_epochs))
            storage.add_attr("batch_size", int(batch_size))
            storage.add_attr("shuffle_data", bool(shuffle_data))
            storage.add_attr("frames_written", int(len(embeddings)))
            storage.add_attr("vocab_size", int(self.vocab_size))
            storage.add_attr("frame_count", int(self.frame_count))
            storage.add_attr("embedding_dtype", str(embeddings[0].dtype))
            storage.add_attr("frame_embeddings_name", block_name)
            storage.add_attr("arrays_per_chunk", int(num_matrices_in_compressed_blocks))
            storage.add_attr("compression_level", int(compression_level))

        _logger.info("Embedding archive written to %s", output_path)
        return str(output_path)

__all__ = ["Embedder"]

if __name__ == "__main__":
    pass
