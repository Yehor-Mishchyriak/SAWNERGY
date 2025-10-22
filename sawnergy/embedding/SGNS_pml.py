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

    def __init__(self,
                V: int,
                D: int,
                *,
                seed: int | None = None,
                optim: Optim,
                optim_kwargs: dict,
                lr_sched: LRScheduler,
                lr_sched_kwargs: dict):
        self.V, self.D = int(V), int(D)
        self.in_emb  = Embedding(V, D)
        self.out_emb = Embedding(V, D)

        self.seed = seed
        self._rng = np.random.default_rng(self.seed)

        self.optim: Optim          = optim(self.parameters, **optim_kwargs)
        self.lr_sched: LRScheduler = lr_sched(**lr_sched_kwargs)

    def _sample_neg(self, B: int, K: int, dist: np.ndarray):
        # (B, K) negatives
        return self._rng.choice(self.V, size=(B, K), replace=True, p=dist)

    def predict(self, center: Tensor, pos: Tensor, neg: Tensor) -> Tensor:
        c      = self.in_emb(center)
        pos_e  = self.out_emb(pos)
        neg_e  = self.out_emb(neg)
        pos_logits = t_sum(c * pos_e, axis=-1)
        neg_logits = t_sum(c[:, None, :] * neg_e, axis=-1)
        #                       ^^^
        # (B,1,D) * (B,K,D) → (B,K,D) → sum D → (B,K) (the None axis is length-1 axis inserted for broadcasting)

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
        data = TensorDataset(centers, contexts)

        for e in range(num_epochs):
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

            if not lr_step_per_batch:
                self.lr_sched.step()

    @property
    def embeddings(self) -> np.ndarray:
        W: Tensor = self.in_emb.parameters[0]
        return W.data


__all__ = ["SGNS_PureML"]

if __name__ == "__main__":
    pass
