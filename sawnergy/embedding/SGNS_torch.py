from __future__ import annotations

# third party
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

# built-in
import logging
from typing import Type
import warnings

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__)

# *----------------------------------------------------*
#                        CLASSES
# *----------------------------------------------------*

class SGNS_Torch:
    """PyTorch implementation of Skip-Gram with Negative Sampling.

    DEPRECATED (temporary): This class currently produces noisy embeddings in
    practice and is deprecated until further notice. The issue likely stems from
    weight initialization, although the root cause has not yet been determined.

    Prefer one of the following alternatives:
      • Plain PyTorch Skip-Gram (full softmax): `SG_Torch`
      • PureML-based implementations: `SGNS_PureML` or `SG_PureML` (if available)

    This API may change or be removed once the root cause is resolved.
    """

    def __init__(self,
                V: int,
                D: int,
                *,
                seed: int | None = None,
                optim: Type[Optimizer] = torch.optim.SGD,
                optim_kwargs: dict | None = None,
                lr_sched: Type[LRScheduler] | None = None,
                lr_sched_kwargs: dict | None = None,
                device: str | None = None):
        """Initialize SGNS (negative sampling) in PyTorch.

        DEPRECATION WARNING:
            This implementation is temporarily deprecated for producing noisy
            embeddings. The issue likely stems from weight initialization, though
            the exact root cause has not been conclusively determined. Please use
            `SG_Torch` (plain Skip-Gram with full softmax) or the PureML-based
            `SGNS_PureML` / `SG_PureML` models instead.

        Args:
            V: Vocabulary size (number of nodes).
            D: Embedding dimensionality.
            seed: Optional RNG seed for PyTorch.
            optim: Optimizer class to instantiate. Defaults to plain SGD.
            optim_kwargs: Keyword arguments for the optimizer. Defaults to {"lr": 0.1}.
            lr_sched: Optional learning-rate scheduler class.
            lr_sched_kwargs: Keyword arguments for the scheduler (required if lr_sched is provided).
            device: Target device string (e.g. "cuda"). Defaults to CUDA if available, else CPU.
        """

        # --- runtime deprecation notice ---
        warnings.warn(
            "SGNS_Torch is temporarily deprecated: it currently produces noisy "
            "embeddings (likely due to weight initialization). Use SG_Torch "
            "(plain Skip-Gram, full softmax) or the PureML-based SG/SGNS classes.",
            DeprecationWarning,
            stacklevel=2,
        )
        _logger.warning(
            "DEPRECATED: SGNS_Torch currently produces noisy embeddings "
            "(likely weight initialization). Prefer SG_Torch or PureML SG/SGNS."
        )
        # ----------------------------------

        optim_kwargs = optim_kwargs or {"lr": 0.1}
        if lr_sched is not None and lr_sched_kwargs is None:
            raise ValueError("lr_sched_kwargs required when lr_sched is provided")

        self.V, self.D = int(V), int(D)
        # two embeddings as in/out matrices
        self.in_emb  = nn.Embedding(self.V, self.D)
        self.out_emb = nn.Embedding(self.V, self.D)

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))

        self.to(self.device)
        _logger.info("SGNS_Torch init: V=%d D=%d device=%s seed=%s", self.V, self.D, self.device, seed)
        params = list(self.in_emb.parameters()) + list(self.out_emb.parameters())
        # optimizer / scheduler
        self.opt = optim(params=params, **optim_kwargs)
        self.lr_sched = lr_sched(self.opt, **lr_sched_kwargs) if lr_sched is not None else None

    def predict(self,
                center: torch.Tensor,
                pos: torch.Tensor,
                neg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        center = center.to(self.device, dtype=torch.long)
        pos    = pos.to(self.device, dtype=torch.long)
        neg    = neg.to(self.device, dtype=torch.long)

        c  = self.in_emb(center)  # (B, D)
        pe = self.out_emb(pos)    # (B, D)
        ne = self.out_emb(neg)    # (B, K, D)

        pos_logits = (c * pe).sum(dim=-1)              # (B,)
        neg_logits = (c.unsqueeze(1) * ne).sum(dim=-1) # (B, K)

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
        """Train SGNS on the provided center/context pairs."""
        if noise_dist.ndim != 1 or noise_dist.size != self.V:
            raise ValueError(f"noise_dist must be 1-D with length {self.V}; got {noise_dist.shape}")
        _logger.info(
            "SGNS_Torch fit: epochs=%d batch=%d negatives=%d shuffle=%s",
            num_epochs, batch_size, num_negative_samples, shuffle_data
        )
        bce = nn.BCEWithLogitsLoss(reduction="mean")

        N = centers.shape[0]
        idx = np.arange(N)

        noise_probs = torch.as_tensor(noise_dist, dtype=torch.float32, device=self.device)
        # require normalized, non-negative distribution
        if (not torch.isfinite(noise_probs).all()
            or (noise_probs < 0).any()
            or abs(float(noise_probs.sum().item()) - 1.0) > 1e-6):
            raise ValueError(
                "noise_dist must be non-negative, finite, and sum to 1.0 "
                f"(got sum={float(noise_probs.sum().item()):.6f}, "
                f"min={float(noise_probs.min().item()):.6f})"
            )

        for epoch in range(1, int(num_epochs) + 1):
            epoch_loss = 0.0
            batches = 0
            if shuffle_data:
                np.random.shuffle(idx)

            for s in range(0, N, int(batch_size)):
                take = idx[s:s+int(batch_size)]
                if take.size == 0:
                    continue
                K = int(num_negative_samples)
                B = len(take)

                cen = torch.as_tensor(centers[take],  dtype=torch.long, device=self.device)  # (B,)
                pos = torch.as_tensor(contexts[take], dtype=torch.long, device=self.device)  # (B,)
                neg = torch.multinomial(noise_probs, num_samples=B * K, replacement=True).view(B, K)  # (B,K) on device

                pos_logits, neg_logits = self(cen, pos, neg)

                # BCE(+)
                y_pos = torch.ones_like(pos_logits)
                loss_pos = bce(pos_logits, y_pos)

                # BCE(-):
                y_neg = torch.zeros_like(neg_logits)
                loss_neg = bce(neg_logits, y_neg)

                loss = loss_pos + K*loss_neg

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()

                if lr_step_per_batch and self.lr_sched is not None:
                    self.lr_sched.step()

                epoch_loss += float(loss.detach().cpu().item())
                batches += 1
                _logger.debug("Epoch %d batch %d loss=%.6f", epoch, batches, loss.item())

            if not lr_step_per_batch and self.lr_sched is not None:
                self.lr_sched.step()

            mean_loss = epoch_loss / max(batches, 1)
            _logger.info("Epoch %d/%d mean_loss=%.6f", epoch, num_epochs, mean_loss)
    
    @property
    def in_embeddings(self) -> np.ndarray:
        W = self.in_emb.weight.detach().cpu().numpy()
        _logger.debug("In emb shape: %s", W.shape)
        return W

    @property
    def out_embeddings(self) -> np.ndarray:
        W = self.out_emb.weight.detach().cpu().numpy()
        _logger.debug("Out emb shape: %s", W.shape)
        return W

    @property
    def avg_embeddings(self) -> np.ndarray:
        return 0.5 * (self.in_embeddings + self.out_embeddings)

    # tiny helper for device move
    def to(self, device):
        self.in_emb.to(device)
        self.out_emb.to(device)
        return self

class SG_Torch:
    """PyTorch implementation of Skip-Gram."""

    def __init__(self,
                V: int,
                D: int,
                *,
                seed: int | None = None,
                optim: Type[Optimizer] = torch.optim.SGD,
                optim_kwargs: dict | None = None,
                lr_sched: Type[LRScheduler] | None = None,
                lr_sched_kwargs: dict | None = None,
                device: str | None = None):
        """Initialize the plain Skip-Gram (full softmax) model in PyTorch.

        Args:
            V: Vocabulary size (number of nodes/tokens).
            D: Embedding dimensionality.
            seed: Optional RNG seed for reproducibility.
            optim: Optimizer class to instantiate. Defaults to :class:`torch.optim.SGD`.
            optim_kwargs: Keyword args for the optimizer. Defaults to ``{"lr": 0.1}``.
            lr_sched: Optional learning-rate scheduler class.
            lr_sched_kwargs: Keyword args for the scheduler (required if ``lr_sched`` is provided).
            device: Target device string (e.g., ``"cuda"``). Defaults to CUDA if available, else CPU.

        Notes:
            The encoder/decoder are linear layers acting on one-hot centers:
            • ``in_emb = nn.Linear(V, D)``
            • ``out_emb = nn.Linear(D, V)``
            Forward pass produces vocabulary-sized logits and is trained with CrossEntropyLoss.
        """
        optim_kwargs = optim_kwargs or {"lr": 0.1}
        if lr_sched is not None and lr_sched_kwargs is None:
            raise ValueError("lr_sched_kwargs required when lr_sched is provided")

        self.V, self.D = int(V), int(D)

        self.in_emb  = nn.Linear(self.V, self.D)
        self.out_emb = nn.Linear(self.D, self.V)

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
        self.to(self.device)
        _logger.info("SG_Torch init: V=%d D=%d device=%s seed=%s", self.V, self.D, self.device, seed)

        params = list(self.in_emb.parameters()) + list(self.out_emb.parameters())
        # optimizer / scheduler
        self.opt = optim(params=params, **optim_kwargs)
        self.lr_sched = lr_sched(self.opt, **lr_sched_kwargs) if lr_sched is not None else None

    def predict(self, center: torch.Tensor) -> torch.Tensor:
        center = center.to(self.device, dtype=torch.long)
        c = nn.functional.one_hot(center, num_classes=self.V).to(dtype=torch.float32, device=self.device)
        y  = self.in_emb(c)
        z = self.out_emb(y)
        return z

    __call__ = predict

    def fit(self,
            centers: np.ndarray,
            contexts: np.ndarray,
            num_epochs: int,
            batch_size: int,
            shuffle_data: bool,
            lr_step_per_batch: bool,
            **_ignore):
        cce = nn.CrossEntropyLoss(reduction="mean")

        N = centers.shape[0]
        idx = np.arange(N)

        for epoch in range(1, int(num_epochs) + 1):
            epoch_loss = 0.0
            batches = 0
            if shuffle_data:
                np.random.shuffle(idx)

            for s in range(0, N, int(batch_size)):
                take = idx[s:s+int(batch_size)]
                if take.size == 0:
                    continue

                cen = torch.as_tensor(centers[take],  dtype=torch.long, device=self.device)
                ctx = torch.as_tensor(contexts[take],  dtype=torch.long, device=self.device)

                logits = self(cen)
                loss = cce(logits, ctx)

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()

                if lr_step_per_batch and self.lr_sched is not None:
                    self.lr_sched.step()

                epoch_loss += float(loss.detach().cpu().item())
                batches += 1
                _logger.debug("Epoch %d batch %d loss=%.6f", epoch, batches, loss.item())

            if not lr_step_per_batch and self.lr_sched is not None:
                self.lr_sched.step()

            mean_loss = epoch_loss / max(batches, 1)
            _logger.info("Epoch %d/%d mean_loss=%.6f", epoch, num_epochs, mean_loss)
    
    @property
    def in_embeddings(self) -> np.ndarray:
        W = self.in_emb.weight.detach().T.cpu().numpy()
        _logger.debug("In emb shape: %s", W.shape)
        return W

    @property
    def out_embeddings(self) -> np.ndarray:
        W = self.out_emb.weight.detach().cpu().numpy()
        _logger.debug("Out emb shape: %s", W.shape)
        return W

    @property
    def avg_embeddings(self) -> np.ndarray:
        return 0.5 * (self.in_embeddings + self.out_embeddings)

    # tiny helper for device move
    def to(self, device):
        self.in_emb.to(device)
        self.out_emb.to(device)
        return self


__all__ = ["SGNS_Torch", "SG_Torch"]

if __name__ == "__main__":
    pass
