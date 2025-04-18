# external imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 keeps 3‑D projection registered
import matplotlib.animation as animation
from typing import Optional, Sequence, Tuple, Union

# local imports
from . import pkg_globals
from . import _util


class NetworkAnalyzer:
    """3‑D visualisation of per‑frame interaction networks.

    Highlights
    ----------
    * **Rainbow colouring** along the polymer chain (residue 1 → residue N).
    * Edge filtering by strongest *p %* (`top_percent`) or absolute cutoff.
    * Zoomed‑in default view (`padding=0.1`).
    * Optionally hide axes (`show_axes=False`).
    * Handles negative weights by plotting |weight|, so edges are never lost.
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(self, network_directory_path: str, config: Optional[dict] = None) -> None:
        # configs -------------------------------------------------------
        self.set_config(pkg_globals.default_config if config is None else config)

        # data ----------------------------------------------------------
        self.residues, self.network_data = _util.import_network_components(
            network_directory_path, self.global_config
        )
        self.number_residues: int = len(self.residues)

        # rainbow colours pre‑computed for every residue ----------------
        cmap = plt.get_cmap("rainbow")
        self.node_colors = cmap(np.linspace(0, 1, self.number_residues))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(config={self.cls_config})"

    def set_config(self, config: dict) -> None:
        self.global_config = config
        self.cls_config = config[self.__class__.__name__]

    # ------------------------------------------------------------------
    # Public visualisers
    # ------------------------------------------------------------------
    def visualize_frame(
        self,
        frame_num: int,
        interaction_type: str | None = None,
        *,
        top_percent: float | None = None,
        figsize: Tuple[int, int] = (10, 8),
        padding: float = 0.1,
        node_size: int = 120,
        edge_scale: float = 1.0,
        show_axes: bool = False,
    ) -> None:
        """Render a single trajectory frame."""
        coords, edges = self._construct_frame(frame_num, interaction_type, top_percent)

        # figure & axes -------------------------------------------------
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        self._set_equal_axes(ax, coords, padding)
        if not show_axes:
            ax.set_axis_off()

        # nodes ---------------------------------------------------------
        x, y, z = coords.T
        ax.scatter(x, y, z, s=node_size, c=self.node_colors, depthshade=True)

        # colour‑bar legend --------------------------------------------
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap="rainbow"),
            ax=ax,
            fraction=0.025,
            pad=0.04,
            label="Residue index (1 → N)",
        )
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["1", str(self.number_residues)])

        # edges ---------------------------------------------------------
        if edges is not None and edges.size:
            for i, j, w in edges:
                xs, ys, zs = coords[[int(i), int(j)]].T
                lw = max(0.5, float(w) * edge_scale)
                ax.plot(xs, ys, zs, linewidth=lw, color="grey", alpha=0.8)

        plt.tight_layout()
        plt.show()

    # ..................................................................
    def visualize_trajectory(
        self,
        frame_range: Union[Sequence[int], Tuple[int, int]],
        interaction_type: str | None = None,
        *,
        top_percent: float | None = None,
        figsize: Tuple[int, int] = (10, 8),
        padding: float = 0.1,
        node_size: int = 120,
        edge_scale: float = 1.0,
        show_axes: bool = False,
        interval: int = 100,
    ) -> animation.FuncAnimation:
        """Animate a segment of the trajectory."""
        # normalise frame list -----------------------------------------
        frames = list(range(frame_range[0], frame_range[1] + 1)) if isinstance(frame_range, tuple) else list(frame_range)

        # global bounding box for steady camera ------------------------
        all_coords = np.concatenate([self._construct_frame(f, None)[0] for f in frames])
        xyz_min, xyz_max = all_coords.min(0), all_coords.max(0)
        span = xyz_max - xyz_min
        xyz_min -= padding * span
        xyz_max += padding * span

        # figure & axes -------------------------------------------------
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(xyz_min[0], xyz_max[0])
        ax.set_ylim(xyz_min[1], xyz_max[1])
        ax.set_zlim(xyz_min[2], xyz_max[2])
        if not show_axes:
            ax.set_axis_off()

        # seed scatter so colour length matches points -----------------
        first_coords, _ = self._construct_frame(frames[0], None)
        scat = ax.scatter(
            first_coords[:, 0], first_coords[:, 1], first_coords[:, 2],
            s=node_size, c=self.node_colors, depthshade=True
        )

        # shared colour‑bar -------------------------------------------
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap="rainbow"),
            ax=ax,
            fraction=0.025,
            pad=0.04,
            label="Residue index (1 → N)",
        )
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["1", str(self.number_residues)])

        # dynamic edge artists -----------------------------------------
        line_handles: list[plt.Artist] = []

        def _clear_lines():
            for h in line_handles:
                h.remove()
            line_handles.clear()

        def update(fnum: int):
            coords, edges = self._construct_frame(fnum, interaction_type, top_percent)
            scat._offsets3d = coords.T
            _clear_lines()
            if edges is not None and edges.size:
                for i, j, w in edges:
                    xs, ys, zs = coords[[int(i), int(j)]].T
                    lw = max(0.5, float(w) * edge_scale)
                    line_handles.append(
                        ax.plot(xs, ys, zs, linewidth=lw, color="grey", alpha=0.8)[0]
                    )
            return scat, *line_handles

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
        plt.tight_layout()
        plt.show()
        return ani

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _construct_frame(
        self,
        frame_num: int,
        interaction_type: str | None = None,
        top_percent: float | None = None,
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """Return coordinates and a filtered edge list."""
        idx = frame_num - 1
        coords = self.network_data[self.cls_config["com_directory_name"]][idx]

        if interaction_type is None:
            return coords, None

        # upper‑triangle interactions ---------------------------------
        mat = self.network_data[interaction_type][0][idx]
        r, c = np.triu_indices(self.number_residues, 1)
        w = np.abs(mat[r, c])  # take magnitude so negatives are visible

        # filtering ----------------------------------------------------
        pct = top_percent if top_percent is not None else self.cls_config.get("weight_cutoff_percent")
        if pct and 0 < pct <= 100 and np.any(w):
            thr = np.percentile(w, 100 - pct)
            mask = w >= thr
        else:
            cutoff = self.cls_config.get("weight_cutoff", 0.0)
            mask = w > cutoff

        edges = np.column_stack((r[mask], c[mask], w[mask]))
        return coords, edges

    # ------------------------------------------------------------------
    @staticmethod
    def _set_equal_axes(ax: Axes3D, coords: np.ndarray, padding: float = 0.1) -> None:
        """Set equal aspect with optional padding."""
        xyz_min, xyz_max = coords.min(0), coords.max(0)
        span = xyz_max - xyz_min
        xyz_min -= padding * span
        xyz_max += padding * span
        ax.set_xlim(xyz_min[0], xyz_max[0])
        ax.set_ylim(xyz_min[1], xyz_max[1])
        ax.set_zlim(xyz_min[2], xyz_max[2])
