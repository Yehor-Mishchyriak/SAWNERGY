from __future__ import annotations

from .embedder import Embedder

def __getattr__(name: str):
    """Lazily expose optional backends."""
    if name == "SGNS_Torch":
        try:
            from .SGNS_torch import SGNS_Torch
        except Exception as exc:
            raise ImportError(
                "PyTorch backend requested but torch is not installed. "
                "Install PyTorch via `pip install torch` (see https://pytorch.org/get-started)."
            ) from exc
        return SGNS_Torch

    if name == "SGNS_PureML":
            try:
                from .SGNS_pml import SGNS_PureML
                return SGNS_PureML
            except Exception as exc:
                raise ImportError(
                    "PureML is not installed. "
                    "Install PureML first via `pip install ym-pure-ml` "
                ) from exc

    raise AttributeError(name)


__all__ = [
    "Embedder",
    "SGNS_PureML",
    "SGNS_Torch",
]
