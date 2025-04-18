# external imports
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 keeps the 3‑D projection registered
import matplotlib.animation as animation
from typing import Optional, Sequence, Tuple, Union

# local imports
from . import pkg_globals
from . import _util


class NetworkAnalyzer:
    """Load per-frame centre-of-mass coordinates and interaction matrices
    (electrostatic, van-der-Waals, hydrogen-bond) and provide static and
    animated 3-D visualisations.

    Parameters
    ----------
    network_directory_path : str
        Folder produced by the preprocessing pipeline. Must contain at least
        *com.npy* plus one or more interaction matrices with the keys present
        in *config*.
    config : dict, optional
        Global config. Falls back to *pkg_globals.default_config*.
    """

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------
    def __init__(self, network_directory_path: str, config: Optional[dict] = None) -> None:
        self.global_config: dict | None = None
        self.cls_config: dict | None = None

        self.set_config(pkg_globals.default_config if config is None else config)

        # Load residues list and a dict‑of‑arrays with all matrices
        # self.network_data["com"]                -> (n_frames, N, 3)
        # self.network_data["elec"][0]            -> (n_frames, N, N)
        self.residues, self.network_data = _util.import_network_components(
            network_directory_path, self.global_config
        )
        self.number_residues: int = len(self.residues)

    # ------------------------------------------------------------------
    # Convenience dunder methods
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover – only for debugging
        return f"{self.__class__.__name__}(config={self.cls_config})"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_config(self, config: dict) -> None:
        """Update run-time configuration and cache subsection for this class."""
        self.global_config = config
        self.cls_config = config[self.__class__.__name__]

    # ..................................................................
    # Single‑frame visualisation
    # ..................................................................
    def visualize_frame(self, frame_num: int, interaction_type: str | None = None) -> None:
        """Render one simulation frame in a blocking Matplotlib window.

        Parameters
        ----------
        frame_num : int
            1-based frame index.
        interaction_type : {"elec", "vdw", "hbond", None}, optional
            Which interaction network to overlay. ``None`` ⇒ points only.
        """
        coords, edges = self._construct_frame(frame_num, interaction_type)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        self._set_equal_axes(ax, coords)

        # --- nodes ------------------------------------------------------
        x, y, z = coords.T
        ax.scatter(x, y, z, s=100, color="blue")
        for idx, (xi, yi, zi) in enumerate(coords):
            ax.text(xi, yi, zi, f"{self.residues[idx]} {idx + 1}", size=10, color="black")

        # --- edges ------------------------------------------------------
        if edges is not None and edges.size:
            for i, j, w in edges:
                xs = coords[[i, j], 0]
                ys = coords[[i, j], 1]
                zs = coords[[i, j], 2]
                ax.plot(xs, ys, zs, linewidth=float(w), color="grey")

        plt.show()

    # ..................................................................
    # Trajectory animation
    # ..................................................................
    def visualize_trajectory(
        self,
        frame_range: Union[Sequence[int], Tuple[int, int]],
        interaction_type: str | None = None,
        interval: int = 100,
    ) -> animation.FuncAnimation:
        """Animate a trajectory segment.

        Parameters
        ----------
        frame_range : sequence[int] | (start, stop)
            Either an explicit list/array of 1-based frames *or* a 2-tuple
            indicating an inclusive range ``(start, stop)``.
        interaction_type : {"elec", "vdw", "hbond", None}, optional
            Which interaction network to draw.
        interval : int, default 100
            Delay between frames in milliseconds.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            The animation handle (useful to save as GIF/MP4).
        """
        # ---- normalise *frame_range* into an iterable ------------------
        if isinstance(frame_range, tuple):
            frames = list(range(frame_range[0], frame_range[1] + 1))
        else:
            frames = list(frame_range)

        # ---- pre‑compute global axis limits for a stable camera --------
        all_coords = np.concatenate(
            [self._construct_frame(f, None)[0] for f in frames], axis=0
        )
        xyz_min = all_coords.min(axis=0) - 2.0
        xyz_max = all_coords.max(axis=0) + 2.0

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(xyz_min[0], xyz_max[0])
        ax.set_ylim(xyz_min[1], xyz_max[1])
        ax.set_zlim(xyz_min[2], xyz_max[2])

        # ---- persistent artists we update in‑place ---------------------
        scat = ax.scatter([], [], [], s=100, color="blue")
        text_handles: list[plt.Text] = []
        line_handles: list[plt.Artist] = []

        def _clear_handles() -> None:
            """Remove old text and lines from the axes."""
            for h in text_handles + line_handles:
                h.remove()
            text_handles.clear()
            line_handles.clear()

        def init():  # noqa: D401 – Matplotlib API
            scat._offsets3d = ([], [], [])
            return (scat,)

        def update(frame_num: int):  # noqa: D401 – Matplotlib API
            coords, edges = self._construct_frame(frame_num, interaction_type)
            x, y, z = coords.T
            scat._offsets3d = (x, y, z)

            _clear_handles()

            # --- node labels ------------------------------------------
            for idx, (xi, yi, zi) in enumerate(coords):
                text_handles.append(
                    ax.text(xi, yi, zi, f"{self.residues[idx]} {idx + 1}", size=10, color="black")
                )

            # --- edges --------------------------------------------------
            if edges is not None and edges.size:
                for i, j, w in edges:
                    xs = coords[[int(i), int(j)], 0]
                    ys = coords[[int(i), int(j)], 1]
                    zs = coords[[int(i), int(j)], 2]
                    line_handles.append(
                        ax.plot(xs, ys, zs, linewidth=float(w), color="grey")[0]
                    )

            return (scat, *text_handles, *line_handles)

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=frames,
            init_func=init,
            interval=interval,
            blit=False,  # blitting is not supported for 3‑D
        )
        plt.show()
        return anim

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _construct_frame(
        self, frame_num: int, interaction_type: str | None = None
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """Vectorised helper that extracts coordinates and edge list.

        Parameters
        ----------
        frame_num : int
            1-based frame index.
        interaction_type : str | None
            Key into *self.network_data* for the interaction matrix. ``None``
            skips edge construction.

        Returns
        -------
        coords : (N, 3) ndarray[float]
            Residue centres of mass.
        edges : (M, 3) ndarray[float] | None
            Array of ``[i, j, weight]`` suitable for fast iteration in the
            caller, or *None* when *interaction_type* is *None*.
        """
        # Convert to 0‑based index expected by the raw arrays
        idx = frame_num - 1
        coords = self.network_data[self.cls_config["com_directory_name"]][idx]

        if interaction_type is None:
            return coords, None

        interaction_matrix = self.network_data[interaction_type][0][idx]
        row, col = np.triu_indices(self.number_residues, k=1)
        weights = interaction_matrix[row, col]

        # Optional cut‑off from config (defaults to zero ⇒ all edges)
        cutoff = self.cls_config.get("weight_cutoff", 0.0)
        mask = weights > cutoff
        edges = np.column_stack((row[mask], col[mask], weights[mask]))
        return coords, edges

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _set_equal_axes(ax: Axes3D, coords: np.ndarray) -> None:
        """Make 3-D axes equally scaled so spheres look like spheres."""
        xyz_min = coords.min(axis=0)
        xyz_max = coords.max(axis=0)
        max_range = (xyz_max - xyz_min).max() / 2
        mid = (xyz_min + xyz_max) / 2
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
