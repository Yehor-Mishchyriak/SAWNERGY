from __future__ import annotations

# third-pary
import numpy as np
# built-in
from pathlib import Path
from typing import Literal
import logging
# local
from . import embedder_util
from .. import sawnergy_util

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__)

# *----------------------------------------------------*
#                        CLASSES
# *----------------------------------------------------*

class Embedder:

    def __init__(self,
                 WALKS_path: str | Path,
                 based_on: Literal["torch", "pureml"],
                 *,
                 seed: int | None = None
                ) -> None:
        _logger.info("Initializing Embedder from %s", WALKS_path)

        # Load numpy arrays from read-only storage
        with sawnergy_util.ArrayStorage(WALKS_path, mode="r") as storage:
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

            num_RWs        = storage.get_attr("num_RWs")
            num_SAWs       = storage.get_attr("num_SAWs")
            node_count     = storage.get_attr("node_count")
            walk_length    = storage.get_attr("walk_length")

        _logger.debug(
            ("Loaded WALKS from %s"
            "| attractive_RWs: shape=%s dtype=%s | "
            "| repulsive_RWs: shape=%s dtype=%s | "
            "| attractive_SAWs: shape=%s dtype=%s | "
            "| repulsive_SAWs: shape=%s dtype=%s | "
            "num_RWs: %d, num_SAWs: %d, node_count: %d, walk_length: %d"),
            WALKS_path,
            getattr(attractive_RWs, "shape", None), getattr(attractive_RWs, "dtype", None),
            getattr(repulsive_RWs, "shape", None), getattr(repulsive_RWs, "dtype", None),
            getattr(attractive_SAWs, "shape", None), getattr(attractive_SAWs, "dtype", None),
            getattr(repulsive_SAWs, "shape", None), getattr(repulsive_SAWs, "dtype", None),
            num_RWs, num_SAWs, node_count, walk_length
        )

        # expected shapes
        RWs_expected_shape  = (num_RWs, walk_length)
        SAWs_expected_shape = (num_SAWs, walk_length)

        # attributes
        self.vocab_size = node_count

        if attractive_RWs is not None:
            if not attractive_RWs.shape == RWs_expected_shape:
                raise ValueError(...)
            self.attractive_RWs = attractive_RWs
        
        if repulsive_RWs is not None:
            if not repulsive_RWs.shape == RWs_expected_shape:
                raise ValueError(...)
            self.repulsive_RWs = repulsive_RWs

        if attractive_SAWs is not None:
            if not attractive_SAWs.shape == SAWs_expected_shape:
                raise ValueError(...)
            self.attractive_SAWs = attractive_SAWs
        
        if repulsive_SAWs is not None:
            if not repulsive_SAWs.shape == SAWs_expected_shape:
                raise ValueError(...)
            self.repulsive_SAWs = repulsive_SAWs

        self.attractive_corpus: np.ndarray | None = None
        self.repulsive_corpus: np.ndarray  | None = None

        # INTERNAL
        self._seed = np.random.randint(0, 2**32 - 1) if seed is None else int(seed)
        self.rng = np.random.default_rng(self._seed)
        _logger.info("RNG initialized from seed=%d", self._seed)

        # SGNS PREREQs:
        # -=-=-=--=-=-=--=-=-=--=-=-=--=-=-=--=-=-=--=-=
        self.soft_unigram:        function | None = None
        self.attractive_corpus: np.ndarray | None = None
        self.repulsive_corpus:  np.ndarray | None = None
        # -=-=-=--=-=-=--=-=-=--=-=-=--=-=-=--=-=-=--=-=

        # SGNS MODEL:
        self.model_base: Literal["torch", "pureml"] = based_on
        self.model: embedder_util.SGNS | None = None

        # SGNS OUTPUT:
        # -=-=-=--=-=-=--=-=-=--=-=-=--=-=-=--=-=-=--=-=
        self.repulsive_embeddings:  np.ndarray | None = None
        self.attractive_embeddings: np.ndarray | None = None
        # -=-=-=--=-=-=--=-=-=--=-=-=--=-=-=--=-=-=--=-=

    def _get_node_frequencies(at: np.ndarray):
        pass

    def _construct_unigram(frequencies: dict):
        pass

    def construct_attractive_corpus(*, using: Literal["RW", "SAW", "merged"], window_size: int):
        pass

    def construct_repulsive_corpus(*, using: Literal["RW", "SAW", "merged"], window_size: int):
        pass

    def embed(*,
            RIN_type: Literal["attr", "repuls"],
            num_negative_samples: int,
            num_epochs: int,
            batch_size: int,
            shuffle: bool,
            dimensionality: int,
            device: str | None = None
        ):
        pass


__all__ = [
    "Embedder"
]

if __name__ == "__main__":
    pass
