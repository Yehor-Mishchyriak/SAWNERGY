from __future__ import annotations

# third party
import numpy as np
from pureml.machinery import Tensor
from pureml.layers import Embedding
from pureml.losses import BCE
from pureml.general_math import sum as t_sum
from pureml.optimizers import Optim, LRScheduler
from pureml.training_utils import TensorDataset, DataLoader
from pureml.base import NN

# built-in
import logging
from typing import Type

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__)

# *----------------------------------------------------*
#                        CLASSES
# *----------------------------------------------------*

class SGNS_PureML(NN):
    """PureML implementation of Skip-Gram with Negative Sampling."""

    def __init__(self,
                V: int,
                D: int,
                *,
                seed: int | None = None,
                optim: Type[Optim],
                optim_kwargs: dict,
                lr_sched: Type[LRScheduler] | None = None,
                lr_sched_kwargs: dict | None = None,
                device: str | None = None):
        """
        Args:
            V: Vocabulary size (number of nodes).
            D: Embedding dimensionality.
            seed: Optional RNG seed for negative sampling.
            optim: Optimizer class to instantiate.
            optim_kwargs: Keyword arguments for the optimizer (required).
            lr_sched: Optional learning-rate scheduler class.
            lr_sched_kwargs: Keyword arguments for the scheduler (required if lr_sched is provided).
            device: Target device string (e.g. "cuda"); accepted for API parity, ignored by PureML.
        """

        if optim_kwargs is None:
            raise ValueError("optim_kwargs must be provided")
        if lr_sched is not None and lr_sched_kwargs is None:
            raise ValueError("lr_sched_kwargs required when lr_sched is provided")

        self.V, self.D = int(V), int(D)

        # embeddings
        self.in_emb  = Embedding(self.V, self.D)
        self.out_emb = Embedding(self.V, self.D)

        # seed + RNG for negative sampling
        self.seed = None if seed is None else int(seed)
        self._rng = np.random.default_rng(self.seed)
        if self.seed is not None:
            # optional: also set global NumPy seed for any non-RNG paths
            np.random.seed(self.seed)

        # API compatibility: PureML is CPU-only
        self.device = "cpu"

        # optimizer / scheduler
        self.optim: Optim = optim(self.parameters, **optim_kwargs)
        self.lr_sched: LRScheduler | None = (
            lr_sched(optim=self.optim, **lr_sched_kwargs) if lr_sched is not None else None
        )

        _logger.info(
            "SGNS_PureML init: V=%d D=%d device=%s seed=%s",
            self.V, self.D, self.device, self.seed
        )

    def _sample_neg(self, B: int, K: int, dist: np.ndarray) -> np.ndarray:
        """Draw negative samples according to the provided unigram distribution."""
        if dist.ndim != 1 or dist.size != self.V:
            raise ValueError(f"noise_dist must be 1-D with length {self.V}; got {dist.shape}")
        return self._rng.choice(self.V, size=(B, K), replace=True, p=dist)

    def predict(self, center: Tensor, pos: Tensor, neg: Tensor) -> tuple[Tensor, Tensor]:
        """Compute positive/negative logits for SGNS.

        Shapes:
            center: (B,)
            pos:    (B,)
            neg:    (B, K)
        Returns:
            pos_logits: (B,)
            neg_logits: (B, K)
        """
        c      = self.in_emb(center)      # (B, D)
        pos_e  = self.out_emb(pos)        # (B, D)
        neg_e  = self.out_emb(neg)        # (B, K, D)

        pos_logits = t_sum(c * pos_e, axis=-1)                # (B,)
        neg_logits = t_sum(c[:, None, :] * neg_e, axis=-1)    # (B, K)
        return pos_logits, neg_logits

    def fit(self,
            centers: np.ndarray,
            contexts: np.ndarray,
            num_epochs: int,
            batch_size: int,
            num_negative_samples: int,
            noise_dist: np.ndarray,
            shuffle_data: bool,
            lr_step_per_batch: bool):
        """Train SGNS on the provided center/context pairs."""
        _logger.info(
            "SGNS_PureML fit: epochs=%d batch=%d negatives=%d shuffle=%s",
            num_epochs, batch_size, num_negative_samples, shuffle_data
        )
        data = TensorDataset(centers, contexts)

        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            batches = 0

            for cen, pos in DataLoader(data, batch_size=batch_size, shuffle=shuffle_data):
                B = cen.data.shape[0] if isinstance(cen, Tensor) else len(cen)

                neg_idx_np = self._sample_neg(B, num_negative_samples, noise_dist)
                neg = Tensor(neg_idx_np, requires_grad=False)
                x_pos_logits, x_neg_logits = self(cen, pos, neg)

                y_pos = Tensor(np.ones_like(x_pos_logits.data))
                y_neg = Tensor(np.zeros_like(x_neg_logits.data))

                loss = (
                    BCE(y_pos, x_pos_logits, from_logits=True)
                    + BCE(y_neg, x_neg_logits, from_logits=True)
                )

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if lr_step_per_batch and self.lr_sched is not None:
                    self.lr_sched.step()

                loss_value = float(np.asarray(loss.data))
                epoch_loss += loss_value
                batches += 1
                _logger.debug("Epoch %d batch %d loss=%.6f", epoch, batches, loss_value)

            if (not lr_step_per_batch) and (self.lr_sched is not None):
                self.lr_sched.step()

            mean_loss = epoch_loss / max(batches, 1)
            _logger.info("Epoch %d/%d mean_loss=%.6f", epoch, num_epochs, mean_loss)

    @property
    def embeddings(self) -> np.ndarray:
        """Return the input embedding matrix as a NumPy array (V, D)."""
        W: Tensor = self.in_emb.parameters[0]
        return np.asarray(W.data)


__all__ = ["SGNS_PureML"]

if __name__ == "__main__":
    pass
