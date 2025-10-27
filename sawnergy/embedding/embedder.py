from __future__ import annotations

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
                 *,
                 seed: int | None = None,
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
            seed: Optional seed for the embedder's RNG. If ``None``, a random
                32-bit seed is chosen.

        Raises:
            ValueError: If required metadata is missing or any loaded walk array
                has an unexpected shape.
            ImportError: If the requested backend is not installed.

        Notes:
            - Walks in storage are 1-based (residue indexing). Internally, this
              class normalizes to 0-based indices for training utilities.
        """
        self._walks_path = Path(WALKS_path)
        _logger.info("Initializing Embedder from %s", self._walks_path)

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

        self.vocab_size   = int(node_count)
        self.frame_count  = int(time_stamp_count)
        self.walk_length  = int(walk_length)
        self.num_RWs      = int(num_RWs)
        self.num_SAWs     = int(num_SAWs)
        # Keep dataset names for metadata passthrough
        self._attractive_RWs_name  = attractive_RWs_name
        self._repulsive_RWs_name   = repulsive_RWs_name
        self._attractive_SAWs_name = attractive_SAWs_name
        self._repulsive_SAWs_name  = repulsive_SAWs_name

        # store walks if present
        if attractive_RWs is not None:
            if RWs_expected and attractive_RWs.shape != RWs_expected:
                raise ValueError(f"ATTR RWs: expected {RWs_expected}, got {attractive_RWs.shape}")
            self.attractive_RWs = attractive_RWs
            _logger.debug("ATTR RWs loaded: %s", self.attractive_RWs.shape)

        if repulsive_RWs is not None:
            if RWs_expected and repulsive_RWs.shape != RWs_expected:
                raise ValueError(f"REP RWs: expected {RWs_expected}, got {repulsive_RWs.shape}")
            self.repulsive_RWs = repulsive_RWs
            _logger.debug("REP  RWs loaded: %s", self.repulsive_RWs.shape)

        if attractive_SAWs is not None:
            if SAWs_expected and attractive_SAWs.shape != SAWs_expected:
                raise ValueError(f"ATTR SAWs: expected {SAWs_expected}, got {attractive_SAWs.shape}")
            self.attractive_SAWs = attractive_SAWs
            _logger.debug("ATTR SAWs loaded: %s", self.attractive_SAWs.shape)

        if repulsive_SAWs is not None:
            if SAWs_expected and repulsive_SAWs.shape != SAWs_expected:
                raise ValueError(f"REP SAWs: expected {SAWs_expected}, got {repulsive_SAWs.shape}")
            self.repulsive_SAWs = repulsive_SAWs
            _logger.debug("REP  SAWs loaded: %s", self.repulsive_SAWs.shape)

        # INTERNAL RNG
        self._seed = np.random.randint(0, 2**32 - 1) if seed is None else int(seed)
        self.rng = np.random.default_rng(self._seed)
        _logger.info("RNG initialized from seed=%d", self._seed)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- PRIVATE -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # HELPERS:

    @staticmethod
    def _get_NN_constructor_from(base: Literal["torch", "pureml"],
                                 objective: Literal["sgns", "sg"]):
        """Resolve the SG/SGNS implementation class for the selected backend.

        Args:
            base: Backend family to use, ``"torch"`` or ``"pureml"``.
            objective: Training objective, ``"sgns"`` or ``"sg"``.

        Returns:
            A callable class (constructor) implementing the requested model.

        Raises:
            ImportError: If the requested backend cannot be imported.
            NameError: If ``base`` is not one of the supported values.
        """
        _logger.debug("Resolving model constructor: base=%s objective=%s", base, objective)
        if base == "torch":
            try:
                from .SGNS_torch import SGNS_Torch, SG_Torch
                ctor = SG_Torch if objective == "sg" else SGNS_Torch
                _logger.debug("Resolved PyTorch class: %s", getattr(ctor, "__name__", str(ctor)))
                return ctor
            except Exception:
                _logger.exception("Failed to import PyTorch backend.")
                raise ImportError(
                    "PyTorch is not installed, but base='torch' was requested. "
                    "Install PyTorch first, e.g.: `pip install torch` "
                    "(see https://pytorch.org/get-started for platform-specific wheels)."
                )
        elif base == "pureml":
            try:
                from .SGNS_pml import SGNS_PureML, SG_PureML
                ctor = SG_PureML if objective == "sg" else SGNS_PureML
                _logger.debug("Resolved PureML class: %s", getattr(ctor, "__name__", str(ctor)))
                return ctor
            except Exception:
                _logger.exception("Failed to import PureML backend.")
                raise ImportError(
                    "PureML is not installed, but base='pureml' was requested. "
                    "Install PureML first via `pip install ym-pure-ml` "
                )
        else:
            raise NameError(f"Expected `base` in (\"torch\", \"pureml\"); Instead got: {base}")

    @staticmethod
    def _as_zerobase_intp(walks: np.ndarray, *, V: int) -> np.ndarray:
        """Validate and convert 1-based walks to 0-based ``intp``.

        Args:
            walks: 2D array of node ids with 1-based indexing.
            V: Vocabulary size for bounds checking.

        Returns:
            2D array of dtype ``intp`` with 0-based indices.

        Raises:
            ValueError: If shape is not 2D or indices are out of bounds.
        """
        arr = np.asarray(walks)
        if arr.ndim != 2:
            raise ValueError("walks must be 2D: (num_walks, walk_len)")
        if arr.dtype.kind not in "iu":
            arr = arr.astype(np.int64, copy=False)
        # 1-based → 0-based
        arr = arr - 1
        mn, mx = int(arr.min()), int(arr.max())
        _logger.debug("Zero-basing walks: min=%d max=%d V=%d", mn, mx, V)
        if mn < 0 or mx >= V:
            raise ValueError("walk ids out of range after 1→0-based normalization")
        return arr.astype(np.intp, copy=False)

    @staticmethod
    def _pairs_from_walks(walks0: np.ndarray, window_size: int) -> np.ndarray:
        """
        Skip-gram pairs including edge centers (one-sided when needed).

        Args:
            walks0: (W, L) int array (0-based ids).
            window_size: Symmetric context window radius.

        Returns:
            Array of shape (N_pairs, 2) int32 with columns [center, context].

        Raises:
            ValueError: If shape is invalid or ``window_size`` <= 0.
        """
        if walks0.ndim != 2:
            raise ValueError("walks must be 2D: (num_walks, walk_len)")

        _, L = walks0.shape
        k = int(window_size)
        _logger.debug("Building SG pairs: L=%d, window=%d", L, k)

        if k <= 0:
            raise ValueError("window_size must be positive")
        
        if L == 0:
            _logger.debug("Empty walks length; returning 0 pairs.")
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
            _logger.debug("No offsets produced pairs; returning empty.")
            return np.empty((0, 2), dtype=np.int32)

        pairs = np.concatenate(out_chunks, axis=0).astype(np.int32, copy=False)
        _logger.debug("Built %d training pairs", pairs.shape[0])
        return pairs

    @staticmethod
    def _freq_from_walks(walks0: np.ndarray, *, V: int) -> np.ndarray:
        """Node frequencies from walks (0-based).

        Args:
            walks0: 2D array of 0-based node ids.
            V: Vocabulary size (minlength for bincount).

        Returns:
            1D array of int64 frequencies with length ``V``.
        """
        freq = np.bincount(walks0.ravel(), minlength=V).astype(np.int64, copy=False)
        _logger.debug("Frequency mass: total=%d nonzero=%d", int(freq.sum()), int(np.count_nonzero(freq)))
        return freq

    @staticmethod
    def _soft_unigram(freq: np.ndarray, *, power: float = 0.75) -> np.ndarray:
        """Return normalized Pn(w) ∝ f(w)^power as float64 probs.

        Args:
            freq: 1D array of token frequencies.
            power: Exponent used for smoothing (default 0.75 à la word2vec).

        Returns:
            1D array of probabilities summing to 1.0.

        Raises:
            ValueError: If mass is invalid (all zeros or non-finite).
        """
        p = np.asarray(freq, dtype=np.float64)
        if p.sum() == 0:
            raise ValueError("all frequencies are zero")
        p = np.power(p, float(power))
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            raise ValueError("invalid unigram mass")
        probs = p / s
        _logger.debug("Noise distribution ready (power=%.3f)", power)
        return probs

    def _materialize_walks(self, frame_id: int, rin: Literal["attr", "repuls"],
                           using: Literal["RW", "SAW", "merged"]) -> np.ndarray:
        """Materialize the requested set of walks for a frame.

        Args:
            frame_id: 1-based frame index.
            rin: Which RIN to pull from: ``"attr"`` or ``"repuls"``.
            using: Which walk sets to include: ``"RW"``, ``"SAW"``, or ``"merged"``.
                If ``"merged"``, concatenate available RW and SAW along axis 0.

        Returns:
            A 2D array of walks with shape (num_walks, walk_length+1).

        Raises:
            IndexError: If ``frame_id`` is out of range.
            ValueError: If no matching walks are available.
        """
        if not 1 <= frame_id <= int(self.frame_count):
            raise IndexError(f"frame_id must be in [1, {self.frame_count}]; got {frame_id}")

        _logger.debug("Materializing %s walks at frame=%d using=%s", rin, frame_id, using)
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
            out = parts[0]
        else:
            out = np.concatenate(parts, axis=0)

        _logger.debug("Materialized walks shape: %s", getattr(out, "shape", None))
        return out

    # INTERFACES: (private)

    def _attractive_corpus_and_prob(self, *,
                                    frame_id: int,
                                    using: Literal["RW", "SAW", "merged"],
                                    window_size: int,
                                    alpha: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
        """Construct (center, context) pairs and noise distribution for ATTR.

        Args:
            frame_id: 1-based frame index.
            using: Walk subset to include.
            window_size: Skip-gram window radius.
            alpha: Unigram smoothing exponent.

        Returns:
            Tuple of (pairs, noise_probs).
        """
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
        """Construct (center, context) pairs and noise distribution for REP.

        Args:
            frame_id: 1-based frame index.
            using: Walk subset to include.
            window_size: Skip-gram window radius.
            alpha: Unigram smoothing exponent.

        Returns:
            Tuple of (pairs, noise_probs).
        """
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
            num_epochs: int,
            negative_sampling: bool = True,
            window_size: int = 2,
            num_negative_samples: int = 10,
            batch_size: int = 1024,
            *,
            lr_step_per_batch: bool = False,
            shuffle_data: bool = True,
            dimensionality: int = 128,
            alpha: float = 0.75,
            device: str | None = None,
            model_base: Literal["torch", "pureml"] = "pureml",
            model_kwargs: dict[str, object] | None = None,
            kind: Literal["in", "out", "avg"] = "in",
            _seed: int | None = None
            ) -> np.ndarray:
        """Train embeddings for a single frame and return the matrix.

        Args:
            frame_id: 1-based frame index to embed.
            RIN_type: ``"attr"`` or ``"repuls"`` - which corpus to use.
            using: Which walks to use (``"RW"``, ``"SAW"``, or ``"merged"``).
            num_epochs: Number of passes over the pairs.
            negative_sampling: If ``True``, use SGNS objective; else plain SG.
            window_size: Skip-gram symmetric window radius.
            num_negative_samples: Negatives per positive pair (SGNS only).
            batch_size: Minibatch size for training.
            lr_step_per_batch: If ``True``, step LR every batch (else per epoch).
            shuffle_data: Shuffle pairs each epoch.
            dimensionality: Embedding dimension ``D``.
            alpha: Unigram smoothing power for noise distribution.
            device: Optional backend device hint (e.g., ``"cuda"``).
            model_base: Backend family (``"torch"`` or ``"pureml"``).
            model_kwargs: Passed through to backend model constructor.
            kind: Which embedding to return: ``"in"``, ``"out"``, or ``"avg"``.
            _seed: Optional override seed for this frame.

        Returns:
            ``(V, D)`` float32 embedding matrix.
        """
        _logger.info(
            "embed_frame: frame=%d RIN=%s using=%s base=%s D=%d epochs=%d batch=%d sgns=%s k=%d alpha=%.3f",
            frame_id, RIN_type, using, model_base, dimensionality, num_epochs, batch_size,
            str(negative_sampling), window_size, alpha
        )

        # ------------------ resolve training data -----------------
        if RIN_type == "attr":
            if self.attractive_RWs is None and self.attractive_SAWs is None:
                raise ValueError("Attractive random walks are missing")
            pairs, noise_probs = self._attractive_corpus_and_prob(frame_id=frame_id, using=using, window_size=window_size, alpha=alpha)
        elif RIN_type == "repuls":
            if self.repulsive_RWs is None and self.repulsive_SAWs is None:
                raise ValueError("Repulsive random walks are missing")
            pairs, noise_probs = self._repulsive_corpus_and_prob(frame_id=frame_id, using=using, window_size=window_size, alpha=alpha)
        else:
            raise NameError(f"Unknown RIN_type: {RIN_type!r}")
        if pairs.size == 0:
            raise ValueError("No training pairs generated for the requested configuration")
        # ----------------------------------------------------------

        # ---------------- construct training corpus ---------------
        centers  = pairs[:, 0].astype(np.int64, copy=False)
        contexts = pairs[:, 1].astype(np.int64, copy=False)
        _logger.debug("Pairs split: centers=%s contexts=%s", centers.shape, contexts.shape)
        # ----------------------------------------------------------

        # ------------ resolve model_constructor kwargs ------------
        if model_kwargs is not None:
            if (("lr_sched" in model_kwargs and model_kwargs.get("lr_sched", None) is not None)
                and ("lr_sched_kwargs" in model_kwargs and model_kwargs.get("lr_sched_kwargs", None) is None)):
                raise ValueError("When `lr_sched`, you must also provide `lr_sched_kwargs`.")

        constructor_kwargs: dict[str, object] = dict(model_kwargs or {})
        constructor_kwargs.update({
            "V": self.vocab_size,
            "D": dimensionality,
            "seed": int(self._seed if _seed is None else _seed),
            "device": device
        })
        _logger.debug("Model constructor kwargs: %s", {k: constructor_kwargs[k] for k in ("V","D","seed","device")})
        # ----------------------------------------------------------

        # --------------- resolve model constructor ----------------
        model_constructor = self._get_NN_constructor_from(
            model_base, objective=("sgns" if negative_sampling else "sg"))
        # ----------------------------------------------------------

        # ------------------ initialize the model ------------------
        model = model_constructor(**constructor_kwargs)
        _logger.debug("Model initialized: %s", model_constructor.__name__ if hasattr(model_constructor,"__name__") else str(model_constructor))
        # ----------------------------------------------------------

        # -------------------- fitting the data --------------------
        _logger.info("Fitting model on %d pairs ...", pairs.shape[0])
        model.fit(centers=centers,
                  contexts=contexts, 
                  num_epochs=num_epochs, 
                  batch_size=batch_size,
                  # -- optional; for SGNS; safely ignored by SG via **_ignore -- 
                  num_negative_samples=num_negative_samples, 
                  noise_dist=noise_probs,
                  # -----------------------------------------
                  shuffle_data=shuffle_data, 
                  lr_step_per_batch=lr_step_per_batch
            )
        _logger.info("Training complete for frame %d", frame_id)
        # ----------------------------------------------------------

        # OUTPUT:
        embeddings = (model.in_embeddings  if kind == "in" else  
                      model.out_embeddings if kind == "out" else
                      model.avg_embeddings if kind == "avg" else
                      None
                )

        if embeddings is None:
            if kind not in ("in", "out", "avg"):
                raise NameError(f"Unknown {kind} embeddings kind. Expected: one of ['in', 'out', 'avg']")

        E = np.asarray(embeddings, dtype=np.float32)
        _logger.debug("Returned embeddings shape: %s dtype=%s", E.shape, E.dtype)
        return E

    def embed_all(
        self,
        RIN_type: Literal["attr", "repuls"],
        using: Literal["RW", "SAW", "merged"],
        num_epochs: int,
        negative_sampling: bool = True,
        window_size: int = 2,
        num_negative_samples: int = 10,
        batch_size: int = 1024,
        *,
        lr_step_per_batch: bool = False,
        shuffle_data: bool = True,
        dimensionality: int = 128,
        alpha: float = 0.75,
        device: str | None = None,
        model_base: Literal["torch", "pureml"] = "pureml",
        model_kwargs: dict[str, object] | None = None,
        kind: Literal["in", "out", "avg"] = "in",
        output_path: str | Path | None = None,
        num_matrices_in_compressed_blocks: int = 20,
        compression_level: int = 3,
        ) -> str:
        """Embed all frames and persist a self-contained archive.

        The resulting file stores a block named ``FRAME_EMBEDDINGS`` with a
        compressed sequence of per-frame matrices (each ``(V, D)``), alongside
        rich metadata mirroring the style of other SAWNERGY modules.

        Args:
            RIN_type: ``"attr"`` or ``"repuls"`` - which corpus to use.
            using: Which walks to use (``"RW"``, ``"SAW"``, or ``"merged"``).
            num_epochs: Number of epochs to train per frame.
            negative_sampling: If ``True``, use SGNS; otherwise plain SG.
            window_size: Skip-gram window radius.
            num_negative_samples: Negatives per positive pair (SGNS).
            batch_size: Minibatch size for training.
            lr_step_per_batch: If ``True``, step LR per batch (else per epoch).
            shuffle_data: Shuffle pairs each epoch.
            dimensionality: Embedding dimension.
            alpha: Unigram smoothing power for noise distribution.
            device: Backend device hint (e.g., ``"cuda"``).
            model_base: Backend family (``"torch"`` or ``"pureml"``).
            model_kwargs: Passed through to backend model constructor.
            kind: Which embedding to store: ``"in"``, ``"out"``, or ``"avg"``.
            output_path: Optional path for the output archive (``.zip`` inferred).
            num_matrices_in_compressed_blocks: How many frames per compressed chunk.
            compression_level: Integer compression level for the archive.

        Returns:
            Path to the created embeddings archive, as ``str``.
        """
        current_time = sawnergy_util.current_time()
        if output_path is None:
            output_path = self._walks_path.with_name(f"EMBEDDINGS_{current_time}").with_suffix(".zip")
        else:
            output_path = Path(output_path)
            if output_path.suffix == "":
                output_path = output_path.with_suffix(".zip")

        _logger.info(
            "embed_all: frames=%d D=%d base=%s RIN=%s using=%s out=%s",
            self.frame_count, dimensionality, model_base, RIN_type, using, output_path
        )

        # Per-frame deterministic seeds
        master_ss = np.random.SeedSequence(self._seed)
        child_seeds = master_ss.spawn(self.frame_count)

        embeddings: list[np.ndarray] = []
        for frame_id, seed_seq in enumerate(child_seeds, start=1):
            child_seed = int(seed_seq.generate_state(1, dtype=np.uint32)[0])
            _logger.info("Embedding frame %d/%d with seed=%d", frame_id, self.frame_count, child_seed)
            E = self.embed_frame(
                    frame_id=frame_id,
                    RIN_type=RIN_type,
                    using=using,
                    num_epochs=num_epochs,
                    negative_sampling=negative_sampling,
                    window_size=window_size,
                    num_negative_samples=num_negative_samples,
                    batch_size=batch_size,
                    lr_step_per_batch=lr_step_per_batch,
                    shuffle_data=shuffle_data,
                    dimensionality=dimensionality,
                    alpha=alpha,
                    device=device,
                    model_base=model_base,
                    model_kwargs=model_kwargs,
                    kind=kind,
                    _seed=child_seed
                )
            embeddings.append(np.asarray(E, dtype=np.float32, copy=False))
            _logger.debug("Frame %d embedded: E.shape=%s", frame_id, E.shape)

        block_name = "FRAME_EMBEDDINGS"
        with sawnergy_util.ArrayStorage.compress_and_cleanup(output_path, compression_level=compression_level) as storage:
            _logger.info("Writing %d frame matrices to block '%s' ...", len(embeddings), block_name)
            storage.write(
                these_arrays=embeddings,
                to_block_named=block_name,
                arrays_per_chunk=num_matrices_in_compressed_blocks
            )

            # Core dataset discovery (for consumers like the Embeddings Visualizer)
            storage.add_attr("frame_embeddings_name", block_name)
            storage.add_attr("time_stamp_count", int(self.frame_count))
            storage.add_attr("node_count", int(self.vocab_size))
            storage.add_attr("embedding_dim", int(dimensionality))

            # Provenance of input WALKS
            storage.add_attr("source_WALKS_path", str(self._walks_path))
            storage.add_attr("walk_length", int(self.walk_length))
            storage.add_attr("num_RWs", int(self.num_RWs))
            storage.add_attr("num_SAWs", int(self.num_SAWs))
            storage.add_attr("attractive_RWs_name", self._attractive_RWs_name)
            storage.add_attr("repulsive_RWs_name",  self._repulsive_RWs_name)
            storage.add_attr("attractive_SAWs_name", self._attractive_SAWs_name)
            storage.add_attr("repulsive_SAWs_name",  self._repulsive_SAWs_name)

            # Training configuration (sufficient to reproduce)
            storage.add_attr("objective", "sgns" if negative_sampling else "sg")
            storage.add_attr("model_base", model_base)
            storage.add_attr("embedding_kind", kind)  # 'in' | 'out' | 'avg'
            storage.add_attr("num_epochs", int(num_epochs))
            storage.add_attr("batch_size", int(batch_size))
            storage.add_attr("window_size", int(window_size))
            storage.add_attr("alpha", float(alpha))
            storage.add_attr("negative_sampling", bool(negative_sampling))
            storage.add_attr("num_negative_samples", int(num_negative_samples))
            storage.add_attr("lr_step_per_batch", bool(lr_step_per_batch))
            storage.add_attr("shuffle_data", bool(shuffle_data))
            storage.add_attr("device_hint", device if device is not None else "")
            storage.add_attr("model_kwargs_repr", repr(model_kwargs) if model_kwargs is not None else "{}")

            # Which walks were used to train
            storage.add_attr("RIN_type", RIN_type)   # 'attr' or 'repuls'
            storage.add_attr("using", using)         # 'RW' | 'SAW' | 'merged'

            # Reproducibility
            storage.add_attr("master_seed", int(self._seed))
            # Note: this records seeds derived from child SeedSequences at metadata time.
            storage.add_attr("per_frame_seeds", [int(s.generate_state(1, dtype=np.uint32)[0]) for s in child_seeds])

            # Archive/IO details
            storage.add_attr("arrays_per_chunk", int(num_matrices_in_compressed_blocks))
            storage.add_attr("compression_level", int(compression_level))
            storage.add_attr("created_at", current_time)

            _logger.info(
                "Stored embeddings archive: %s | shape=(T,N,D)=(%d,%d,%d)",
                output_path, self.frame_count, self.vocab_size, dimensionality
            )

        return str(output_path)


__all__ = ["Embedder"]

if __name__ == "__main__":
    pass
