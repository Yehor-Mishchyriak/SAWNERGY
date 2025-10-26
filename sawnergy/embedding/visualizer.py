from __future__ import annotations

# third party
import numpy as np
import matplotlib as mpl

# built-in
from pathlib import Path
from typing import Sequence
import logging

# local
from ..visual import visualizer_util
from .. import sawnergy_util

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__)

# *----------------------------------------------------*
#                        HELPERS
# *----------------------------------------------------*

def _safe_svd_pca(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute k principal directions via SVD and project onto them."""
    if X.ndim != 2:
        raise ValueError(f"PCA expects 2D array (N, D); got {X.shape}")
    _, D = X.shape
    if k not in (2, 3):
        raise ValueError(f"PCA dimensionality must be 2 or 3; got {k}")
    if D < k:
        raise ValueError(f"Requested k={k} exceeds feature dim D={D}")
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:k].copy()
    proj = Xc @ comps.T
    return proj, comps

def _set_equal_axes_3d(ax, xyz: np.ndarray, *, padding: float = 0.05) -> None:
    if xyz.size == 0:
        return
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    zmin, zmax = float(z.min()), float(z.max())
    xr = xmax - xmin
    yr = ymax - ymin
    zr = zmax - zmin
    r = max(xr, yr, zr)
    pad = padding * (r if r > 0 else 1.0)
    cx, cy, cz = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0
    ax.set_xlim(cx - r / 2 - pad, cx + r / 2 + pad)
    ax.set_ylim(cy - r / 2 - pad, cy + r / 2 + pad)
    ax.set_zlim(cz - r / 2 - pad, cz + r / 2 + pad)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass

# *----------------------------------------------------*
#                        CLASS
# *----------------------------------------------------*

class Visualizer:
    """3D PCA visualizer for per-frame embeddings"""

    no_instances: bool = True

    def __init__(
        self,
        EMB_path: str | Path,
        figsize: tuple[int, int] = (9, 7),
        default_node_color: str = visualizer_util.GRAY,
        depthshade: bool = False,
        antialiased: bool = False,
        init_elev: float = 35,
        init_azim: float = 45,
        *,
        show: bool = False
    ) -> None:
        # Backend & pyplot
        visualizer_util.ensure_backend(show)
        import matplotlib.pyplot as plt
        self._plt = plt

        if Visualizer.no_instances:
            try:
                visualizer_util.warm_start_matplotlib()
            finally:
                Visualizer.no_instances = False

        # Load embeddings archive
        EMB_path = Path(EMB_path)
        with sawnergy_util.ArrayStorage(EMB_path, mode="r") as storage:
            name = storage.get_attr("frame_embeddings_name")
            E = storage.read(name, slice(None))
        if E.ndim != 3:
            raise ValueError(f"Expected embeddings of shape (T,N,D); got {E.shape}")
        self.E = np.asarray(E)
        self.T, self.N, self.D = map(int, self.E.shape)
        _logger.info("Loaded embeddings: T=%d, N=%d, D=%d", self.T, self.N, self.D)

        # Coloring normalizer (parity with RIN Visualizer)
        self._residue_norm = mpl.colors.Normalize(0, max(1, self.N - 1))

        # Figure / axes / artists
        self._fig = self._plt.figure(figsize=figsize)
        self._ax = None
        self._scatter = None
        self._marker_size = 30.0
        self._init_elev = init_elev
        self._init_azim = init_azim
        self.default_node_color = default_node_color
        self._antialiased = bool(antialiased)
        self._depthshade = bool(depthshade)

    # ------------------------------ PRIVATE ------------------------------ #

    def _ensure_axes(self) -> None:
        if self._ax is not None and self._scatter is not None:
            return
        self._fig.clf()
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._ax.view_init(elev=self._init_elev, azim=self._init_azim)
        self._scatter = self._ax.scatter(
            [], [], [],
            s=self._marker_size,
            depthshade=self._depthshade,
            edgecolors="none",
            antialiased=self._antialiased,
        )
        try:
            self._ax.set_axis_off()
        except Exception:
            pass

    def _project3(self, X: np.ndarray) -> np.ndarray:
        """Return a 3D PCA projection of embeddings (always 3 coordinates).

        If the embedding dimensionality D < 3, the remaining coordinate(s) are set to 0
        so that the returned array still has shape (N, 3).
        """
        k = 3 if X.shape[1] >= 3 else 2
        P, _ = _safe_svd_pca(X, k)
        if k == 2:
            P = np.c_[P, np.zeros((P.shape[0], 1), dtype=P.dtype)]
        return P

    def _select_nodes(self, displayed_nodes: Sequence[int] | str | None) -> np.ndarray:
        if displayed_nodes is None or displayed_nodes == "ALL":
            return np.arange(self.N, dtype=np.int64)
        idx = np.asarray(displayed_nodes)
        if idx.dtype.kind not in "iu":
            raise TypeError("displayed_nodes must be None, 'ALL', or an integer sequence.")
        if idx.min() < 1 or idx.max() > self.N:
            raise IndexError(f"displayed_nodes out of range [1,{self.N}]")
        return idx.astype(np.int64) - 1

    def _apply_colors(self, node_colors, idx: np.ndarray) -> np.ndarray:
        # RIN Visualizer semantics:
        if isinstance(node_colors, str):
            node_cmap = self._plt.get_cmap(node_colors)
            return node_cmap(self._residue_norm(idx))
        if node_colors is None:
            full = visualizer_util.map_groups_to_colors(
                N=self.N, groups=None, default_color=self.default_node_color, one_based=True
            )
            return np.asarray(full)[idx]
        arr = np.asarray(node_colors)
        if arr.ndim == 2 and arr.shape[0] == self.N and arr.shape[1] in (3, 4):
            return arr[idx]
        full = visualizer_util.map_groups_to_colors(
            N=self.N, groups=node_colors, default_color=self.default_node_color, one_based=True
        )
        return np.asarray(full)[idx]

    # ------------------------------ PUBLIC ------------------------------- #

    def build_frame(
        self,
        frame_id: int,
        *,
        node_colors: str | np.ndarray | None = "rainbow",
        displayed_nodes: Sequence[int] | str | None = "ALL",
        show_node_labels: bool = False,
        show: bool = False
    ) -> None:
        """Render a single frame as a PCA **3D** scatter (matches RIN Visualizer API)."""
        frame0 = int(frame_id) - 1
        if not (0 <= frame0 < self.T):
            raise IndexError(f"frame_id out of range [1,{self.T}]")
        self._ensure_axes()

        idx = self._select_nodes(displayed_nodes)
        X = self.E[frame0, idx, :]   # (n, D)
        P = self._project3(X)        # (n, 3)
        colors = self._apply_colors(node_colors, idx)

        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        self._scatter._offsets3d = (x, y, z)
        self._scatter.set_facecolors(colors)
        _set_equal_axes_3d(self._ax, P, padding=0.05)
        self._ax.view_init(elev=self._init_elev, azim=self._init_azim)

        if show_node_labels:
            for txt in getattr(self, "_labels", []):
                try:
                    txt.remove()
                except Exception:
                    pass
            self._labels = []
            for p, nid in zip(P, idx + 1):
                self._labels.append(self._ax.text(p[0], p[1], p[2], str(int(nid)), fontsize=8))

        self._fig.tight_layout()
        self._fig.canvas.draw_idle()

        if show:
            try:
                self._plt.show(block=True)
            except TypeError:
                self._plt.show()

    # convenience
    def savefig(self, path: str | Path, *, dpi: int = 150) -> None:
        self._fig.savefig(path, dpi=dpi)

    def close(self) -> None:
        try:
            self._plt.close(self._fig)
        except Exception:
            pass


__all__ = ["Visualizer"]
