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

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- PRIVATE -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # HELPERS:

    @staticmethod
    def _get_NN_constructor_from(base: Literal["torch", "pureml"],
                                 objective: Literal["sgns", "sg"]):
        """Resolve the SG/SGNS implementation class for the selected backend."""
        if base == "torch":
            try:
                from .SGNS_torch import SGNS_Torch, SG_Torch
                return SG_Torch if objective == "sg" else SGNS_Torch
            except Exception:
                raise ImportError(
                    "PyTorch is not installed, but base='torch' was requested. "
                    "Install PyTorch first, e.g.: `pip install torch` "
                    "(see https://pytorch.org/get-started for platform-specific wheels)."
                )
        elif base == "pureml":
            try:
                from .SGNS_pml import SGNS_PureML, SG_PureML
                return SG_PureML if objective == "sg" else SGNS_PureML
            except Exception:
                raise ImportError(
                    "PureML is not installed, but base='pureml' was requested. "
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
        # ----------------------------------------------------------

        # ------------ resolve model_constructor kwargs ------------
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
        # ----------------------------------------------------------

        # --------------- resolve model constructor ----------------
        model_constructor = self._get_NN_constructor_from(
            model_base, objective=("sgns" if negative_sampling else "sg"))
        # ----------------------------------------------------------

        # ------------------ initialize the model ------------------
        model = model_constructor(**constructor_kwargs)
        # ----------------------------------------------------------

        # -------------------- fitting the data --------------------
        model.fit(centers=centers,
                  contexts=contexts, 
                  num_epochs=num_epochs, 
                  batch_size=batch_size,
                  # -- optional: ----------------------------
                  num_negative_samples=num_negative_samples, 
                  noise_dist=noise_probs,
                  # -----------------------------------------
                  shuffle_data=shuffle_data, 
                  lr_step_per_batch=lr_step_per_batch
            )
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

        return np.asarray(embeddings)

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
        ):

        current_time = sawnergy_util.current_time()
        if output_path is None:
            output_path = self._walks_path.with_name(f"EMBEDDINGS_{current_time}").with_suffix(".zip")
        else:
            output_path = Path(output_path)
            if output_path.suffix == "":
                output_path = output_path.with_suffix(".zip")

        master_ss = np.random.SeedSequence(self._seed)
        child_seeds = master_ss.spawn(self.frame_count)

        embeddings = []
        for frame_id, seed_seq in enumerate(child_seeds, start=1):
            child_seed = int(seed_seq.generate_state(1, dtype=np.uint32)[0])
            embeddings.append(
                self.embed_frame(
                    frame_id,
                    RIN_type,
                    using,
                    num_epochs,
                    negative_sampling,
                    window_size,
                    num_negative_samples,
                    batch_size,
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
            )

        block_name = "FRAME_EMBEDDINGS"
        with sawnergy_util.ArrayStorage.compress_and_cleanup(output_path, compression_level=compression_level) as storage:
            storage.write(
                these_arrays=embeddings,
                to_block_named=block_name,
                arrays_per_chunk=num_matrices_in_compressed_blocks
            )
            # ... add metadata

        return str(output_path)


__all__ = ["Embedder"]

if __name__ == "__main__":
    pass
