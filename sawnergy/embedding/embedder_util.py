# embedder_util.py
import numpy as np
from pureml.machinery import Tensor
from pureml.layers import Embedding
from pureml.losses import BCE
from pureml.general_math import sum as t_sum
from pureml.optimizers import Adam
from pureml.training_utils import TensorDataset
from pureml.base import NN

class SGNS(NN):

    def __init__(self, V: int, D: int, *, seed: int | None = None):
        self.V, self.D = int(V), int(D)
        self.in_emb  = Embedding(V, D)
        self.out_emb = Embedding(V, D)

        self.seed = seed
        self._rng = np.random.default_rng(self.seed)

    def _sample_neg(self, B, K, dist: np.ndarray):
        # (B, K) negatives
        return self._rng.choice(self.V, size=(B, K), replace=True, p=dist)

    def predict(self, center: Tensor, pos: Tensor, neg: Tensor) -> Tensor:
        c      = self.in_emb(center)
        pos_e  = self.out_emb(pos)
        neg_e  = self.out_emb(neg)
        pos_logits = t_sum(c * pos_e, axis=-1)
        neg_logits = t_sum(c[:, None, :] * neg_e, axis=-1) 

        return pos_logits, neg_logits

    def fit(centers, contexts, num_epochs, batch_size, num_negative_samples, shuffle_data):
        data = TensorDataset(centers, contexts)

        # Lookups
        c      = self.in_emb(center)                 # (B, D)
        pos_e  = self.out_emb(pos)                   # (B, D)
        neg_e  = self.out_emb(neg)                   # (B, K, D)

        # Dot products → logits
        pos_logits = t_sum(c * pos_e, axis=-1)               # (B,)
        neg_logits = t_sum(c[:, None, :] * neg_e, axis=-1)   # (B, K)

        # Targets
        y_pos = Tensor(np.ones_like(pos_logits.data))
        y_neg = Tensor(np.zeros_like(neg_logits.data))

        # Logistic loss (averaged internally)
        loss = BCE(y_pos, pos_logits, from_logits=True) + BCE(y_neg, neg_logits, from_logits=True)

        # Backprop + step
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return float(loss.data)  # scalar
