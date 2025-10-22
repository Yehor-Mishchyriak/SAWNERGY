from __future__ import annotations

# third party
import numpy as np
import torch
import torch.nn as nn

# built-in
from typing import Any, Literal
import logging

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__)

# *----------------------------------------------------*
#                        CLASSES
# *----------------------------------------------------*

class SGNS_Torch:

    def __init__(self,
                 V: int,
                 D: int,
                 *,
                seed: int | None = None,
                optim: Optim,
                optim_kwargs: dict,
                lr_sched: LRScheduler,
                lr_sched_kwargs: dict,
                device: str):

        self.V, self.D = int(V), int(D)
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        if seed is not None:
            torch.manual_seed(int(seed))

        # two embeddings as in/out matrices
        self.in_emb  = nn.Embedding(self.V, self.D)
        self.out_emb = nn.Embedding(self.V, self.D)

        # simple optimizer; schedulers can be layered externally if you like
        self.opt = torch.optim.Adam(list(self.in_emb.parameters()) + list(self.out_emb.parameters()), lr=lr)

        self.to(self.device)

    # keep the same name used by PureML's NN (predict) so Embedder can call self(...)
    def predict(self, center: np.ndarray, pos: np.ndarray, neg: np.ndarray):
        # center: (B,), pos: (B,), neg: (B,K)
        c  = self.in_emb(torch.as_tensor(center, dtype=torch.long, device=self.device))          # (B, D)
        pe = self.out_emb(torch.as_tensor(pos,     dtype=torch.long, device=self.device))        # (B, D)
        ne = self.out_emb(torch.as_tensor(neg,     dtype=torch.long, device=self.device))        # (B, K, D)

        pos_logits = (c * pe).sum(dim=-1)                               # (B,)
        neg_logits = (c.unsqueeze(1) * ne).sum(dim=-1)                  # (B, K)
        return pos_logits, neg_logits

    __call__ = predict

    def fit(self,
            centers: np.ndarray,
            contexts: np.ndarray,
            num_epochs: int,
            batch_size: int,
            num_negative_samples: int,
            noise_dist: np.ndarray,
            shuffle_data: bool,
            lr_step_per_batch: bool):
        _require_torch()
        
        bce = nn.BCEWithLogitsLoss(reduction="mean")

        N = centers.shape[0]
        idx = np.arange(N)

        # torch noise distribution once
        noise_probs = torch.as_tensor(noise_dist, dtype=torch.float32, device=self.device)

        for _ in range(int(num_epochs)):
            if shuffle_data:
                np.random.shuffle(idx)

            for s in range(0, N, int(batch_size)):
                take = idx[s:s+int(batch_size)]
                if take.size == 0:
                    continue

                cen = centers[take]
                pos = contexts[take]
                # negatives: sample with replacement using torch (GPU friendly)
                neg = torch.multinomial(noise_probs, num_samples=int(num_negative_samples * take.size), replacement=True)
                neg = neg.view(len(take), int(num_negative_samples)).detach().cpu().numpy()

                pos_logits, neg_logits = self(cen, pos, neg)

                # BCE(+)
                y_pos = torch.ones_like(pos_logits)
                loss_pos = bce(pos_logits, y_pos)

                # BCE(-): flatten (B,K) -> (B*K,)
                y_neg = torch.zeros_like(neg_logits)
                loss_neg = bce(neg_logits, y_neg)

                loss = loss_pos + loss_neg

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()

        # (no scheduler here; add if you want)

    @property
    def parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        return (_to_numpy(self.in_emb.weight), _to_numpy(self.out_emb.weight))

    # tiny helper for device move
    def to(self, device):
        self.in_emb.to(device)
        self.out_emb.to(device)
        return self


__all__ = ["SGNS_PureML"]

if __name__ == "__main__":
    pass
