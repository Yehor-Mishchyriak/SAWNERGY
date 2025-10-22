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
                optim: Optim,
                optim_kwargs: dict,
                lr_sched: LRScheduler,
                lr_sched_kwargs: dict):
        """
        Args:
            V: Vocabulary size (number of nodes).
            D: Embedding dimensionality.
            seed: Optional RNG seed for negative sampling.
            optim: PureML optimizer class.
            optim_kwargs: Keyword arguments forwarded to the optimizer.
            lr_sched: PureML learning-rate scheduler class.
            lr_sched_kwargs: Keyword arguments forwarded to the scheduler.
        """
        self.V, self.D = int(V), int(D)
        self.in_emb  = Embedding(V, D)
        self.out_emb = Embedding(V, D)

        self.seed = None if seed is None else int(seed)
        self._rng = np.random.default_rng(self.seed)

        self.optim: Optim          = optim(self.parameters, **optim_kwargs)
        self.lr_sched: LRScheduler = lr_sched(**lr_sched_kwargs)
        _logger.info("SGNS_PureML init: V=%d D=%d seed=%s", self.V, self.D, self.seed)

    def _sample_neg(self, B: int, K: int, dist: np.ndarray):
        """Draw negative samples according to the provided unigram distribution."""
        if dist.ndim != 1 or dist.size != self.V:
            raise ValueError(f"noise_dist must be 1-D with length {self.V}; got {dist.shape}")
        return self._rng.choice(self.V, size=(B, K), replace=True, p=dist)

    def predict(self, center: Tensor, pos: Tensor, neg: Tensor) -> Tensor:
        """Compute positive/negative logits for SGNS."""
        c      = self.in_emb(center)
        pos_e  = self.out_emb(pos)
        neg_e  = self.out_emb(neg)
        pos_logits = t_sum(c * pos_e, axis=-1)
        neg_logits = t_sum(c[:, None, :] * neg_e, axis=-1)
        #                       ^^^
        # (B,1,D) * (B,K,D) → (B,K,D) → sum D → (B,K)

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
                neg = self._sample_neg(batch_size, num_negative_samples, noise_dist)

                x_pos_logits, x_neg_logits = self(cen, pos, neg)

                y_pos = Tensor(np.ones_like(x_pos_logits.data))
                y_neg = Tensor(np.zeros_like(x_neg_logits.data))

                loss = BCE(y_pos, x_pos_logits, from_logits=True) + BCE(y_neg, x_neg_logits, from_logits=True)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                if lr_step_per_batch:
                    self.lr_sched.step()

                loss_value = float(np.asarray(loss.data).mean())
                epoch_loss += loss_value
                batches += 1
                _logger.debug("Epoch %d batch %d loss=%.6f", epoch, batches, loss_value)

            if not lr_step_per_batch:
                self.lr_sched.step()

            mean_loss = epoch_loss / max(batches, 1)
            _logger.info("Epoch %d/%d mean_loss=%.6f", epoch, num_epochs, mean_loss)

    @property
    def embeddings(self) -> np.ndarray:
        """Return the input embedding matrix as a NumPy array."""
        W: Tensor = self.in_emb.parameters[0]
        return np.asarray(W.data)


__all__ = ["SGNS_PureML"]

if __name__ == "__main__":
    pass
