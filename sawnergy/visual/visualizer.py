from __future__ import annotations

# third-pary
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
# built-in
from pathlib import Path
from typing import Iterable
import logging
# local
from .. import sawnergy_util
from . import visualizer_util

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__) 

# DEFAULT COLORS:
BLUE = "#3B82F6"        # Tailwind Blue 500
GREEN = "#10B981"       # Emerald Green
RED = "#EF4444"         # Soft Red
YELLOW = "#FACC15"      # Amber Yellow
PURPLE = "#8B5CF6"      # Vibrant Purple
PINK = "#EC4899"        # Modern Pink
TEAL = "#14B8A6"        # Teal
ORANGE = "#F97316"      # Bright Orange
CYAN = "#06B6D4"        # Cyan
INDIGO = "#6366F1"      # Indigo
GRAY = "#6B7280"        # Neutral Gray
LIME = "#84CC16"        # Lime Green
ROSE = "#F43F5E"        # Rose
SKY = "#0EA5E9"         # Sky Blue
SLATE = "#475569"       # Slate Gray

# *----------------------------------------------------*
#                        CLASSES
# *----------------------------------------------------*

class Visualizer:
    
    def __init__(self,
                RIN_path: str | Path,
                COM_dataset_name: str = "COM",
                attr_data_name: str = "ATTRACTIVE_energies",
                repuls_data_name: str = "REPULSIVE_energies") -> None:
        
        self.RIN_data = sawnergy_util.ArrayStorage(RIN_path, mode="r")
        self.COM_data_name: str = COM_dataset_name
        self.attr_data_name: str = attr_data_name
        self.repuls_data_name: str = repuls_data_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.RIN_data.close()

    def __del__(self):
        try:
            self.RIN_data.close()
        except Exception:
            pass

    # --------- PRIVATE ----------

    def _construct_frame(
        self,
        axes: Axes3D,
        coordinates: np.ndarray,
        attractive_interactions: np.ndarray | None,
        repulsive_interactions: np.ndarray | None,
        node_size: int,
        edge_scale: float,
        top_percent_displayed: float,
        node_colors: str | dict[Iterable[int], str] | None,
        attractive_edge_color: str,
        repulsive_edge_color: str,
        default_color: str,
        padding: float,
        elev: float,
        azim: float,
        show_axes: bool,
        one_based: bool
    ):
        
        # VALIDATE COORDS
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError(f"`coordinates` must be (N, 3); got {coordinates.shape}")
        N = coordinates.shape[0]

        # EXPAND THE AXES TO FIT DATA WELL
        xyz_min = coordinates.min(axis=0)
        xyz_max = coordinates.max(axis=0)
        span = np.maximum(xyz_max - xyz_min, 1e-12) # avoid zero-span
        xyz_min -= padding * span
        xyz_max += padding * span
        axes.set_xlim(xyz_min[0], xyz_max[0])
        axes.set_ylim(xyz_min[1], xyz_max[1])
        axes.set_zlim(xyz_min[2], xyz_max[2])
        axes.view_init(elev=elev, azim=azim)
        if not show_axes: # optionally remove the axes
            axes.set_axis_off()

        # COLOR THE DATA POINTS
        if isinstance(node_colors, str):
            cmap = plt.get_cmap(node_colors)
            idx = np.arange(1, N + 1) if one_based else np.arange(N)
            norm = mpl.colors.Normalize(vmin=idx.min(), vmax=idx.max())
            color_array = cmap(norm(idx))
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = axes.figure.colorbar(sm, ax=axes, fraction=0.025, pad=0.04)
            cbar.set_label(f"Residue index ({'1' if one_based else '0'} → {N if one_based else N-1})")
            cbar.set_ticks([idx.min(), idx.max()])
        else:
            if node_colors is None:
                color_array = np.array([mpl.colors.to_rgba(default_color)] * N)
            else:
                color_array = visualizer_util.map_groups_to_colors(
                    N=N, groups=node_colors, default_color=default_color, one_based=one_based
                )

        # PLOT THE DATA POINTS (RESIDUES)
        axes.scatter(
            xs=coordinates[:, 0], ys=coordinates[:, 1], zs=coordinates[:, 2],
            s=node_size, c=color_array, depthshade=True
        )

        # ADD INTERACTIONS DATA IF AVAILABLE
        if attractive_interactions is not None:
            Ath = np.quantile(attractive_interactions, 1 - top_percent_displayed)
            attr_edges = visualizer_util.create_weighted_edge_collection(
                                coordinates,
                                attractive_interactions,
                                Ath,
                                edge_scale,
                                colors=attractive_edge_color)
            axes.add_collection3d(attr_edges)

        if repulsive_interactions is not None:
            Rth = np.quantile(repulsive_interactions, 1 - top_percent_displayed)
            repuls_edges = visualizer_util.create_weighted_edge_collection(
                                coordinates,
                                repulsive_interactions,
                                Rth,
                                edge_scale,
                                colors=repulsive_edge_color)
            axes.add_collection3d(repuls_edges)
        
        rng = np.maximum(xyz_max - xyz_min, 1e-12) # data range
        axes.set_box_aspect(rng)
        # ^ from the docs: 'set the box aspect to match your data range in each dimension'

    # --------- PUBLIC ----------
    def visualize_frame(
        self,
        frame_id: int,
        display_attractive_interactions: bool = True,
        display_repulsive_interactions: bool = True,
        node_size: int = 15,
        edge_scale: float = 1,
        top_percent_displayed: float = 0.1,
        node_colors: str | dict[Iterable[int], str] | None = None,
        one_based_residues: bool = True,
        attractive_edge_color: str = GREEN,
        repulsive_edge_color: str = RED,
        default_color: str = GRAY,
        title: str | None = None,
        figsize: tuple[int, int] = (8, 6),
        padding: float = 0.5,
        elev: float = 35,
        azim: float = 45,
        show_axes: bool = False,
        show: bool = False
    ) -> None:
    
        if not (0.0 < top_percent_displayed <= 1.0):
            raise ValueError("top_percent_displayed must be in (0, 1].")

        coordinates = self.RIN_data.read(
            from_block_named = self.COM_data_name,
            ids = frame_id
        )

        attractive_interactions = self.RIN_data.read(
            from_block_named = self.attr_data_name,
            ids = frame_id
        ) if display_attractive_interactions else None

        repulsive_interactions = self.RIN_data.read(
            from_block_named = self.repuls_data_name,
            ids = frame_id
        ) if display_repulsive_interactions else None
    
        fig = plt.figure(figsize=figsize)

        ax: Axes3D = fig.add_subplot(111, projection="3d")
        if title:
            fig.suptitle(title)

        self._construct_frame(
                ax,
                coordinates,
                attractive_interactions,
                repulsive_interactions,
                node_size,
                edge_scale,
                top_percent_displayed,
                node_colors,
                attractive_edge_color,
                repulsive_edge_color,
                default_color,
                padding,
                elev,
                azim,
                show_axes,
                one_based_residues
            )
        
        if show:
            plt.show()


__all__ = [
    "Visualizer",
    "BLUE",
    "GREEN",
    "RED",
    "YELLOW",
    "PURPLE",
    "PINK",
    "TEAL",
    "ORANGE",
    "CYAN",
    "INDIGO",
    "GRAY",
    "LIME",
    "ROSE",
    "SKY",
    "SLATE"
]

if __name__ == "__main__":
    pass
