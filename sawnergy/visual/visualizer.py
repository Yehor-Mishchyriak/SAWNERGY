from __future__ import annotations

# third-pary
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import PathCollection
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

# *----------------------------------------------------*
#                        CLASSES
# *----------------------------------------------------*

class Visualizer:

    no_instances: bool = True

    def __init__(
        self,
        RIN_path: str | Path,
        figsize: tuple[int, int] = (8, 6),
        node_size: int = 120,
        edge_width: float = 0.5,
        default_node_color: str = visualizer_util.GRAY,
        depthshade: bool = False,
        antialiased: bool = False,
        *,
        COM_dataset_name: str = "COM",
        attr_data_name: str = "ATTRACTIVE_energies",
        repuls_data_name: str = "REPULSIVE_energies",
    ) -> None:

    # ---------- WARM UP MPL ------------ #
        if Visualizer.no_instances:
            visualizer_util.warm_start_matplotlib()
    
    # ---------- LOAD THE DATA ---------- #
        with sawnergy_util.ArrayStorage(RIN_path, mode="r") as storage:
            self.COM_coords: str      = storage.read(COM_dataset_name, slice(None))
            self.attr_energies: str   = storage.read(attr_data_name, slice(None))
            self.repuls_energies: str = storage.read(repuls_data_name, slice(None))

        N = np.size(self.COM_coords[0], axis=0)

    # - SET UP THE CANVAS AND THE AXES - #
        self._fig = plt.figure(figsize=figsize)
        self._ax  = self._fig.add_subplot(111, projection="3d")
        self._ax.set_autoscale_on(False)
    
    # ------ SET UP PLOT ELEMENTS ------ #
        self._scatter: PathCollection  = self._ax.scatter([], [], [], s=node_size, c=default_node_color, depthshade=depthshade, edgecolors="none")
        self._attr: Line3DCollection   = Line3DCollection(np.empty((0,2,3)), linewidths=edge_width, antialiased=antialiased)
        self._repuls: Line3DCollection = Line3DCollection(np.empty((0,2,3)), linewidths=edge_width, antialiased=antialiased)
        self._ax.add_collection3d(self._attr); self._ax.add_collection3d(self._repuls) # set pointers to the attractive and repulsive collections

    # ---------- HELPER FIELDS --------- #
        # NOTE: 'under the hood' everything is 0-base indexed,
        # BUT, from the API point of view, the idexing is 1-base,
        # because amino acid residues are 1-base indexed.
        self._residue_norm = mpl.colors.Normalize(0, N-1) # uniform coloring

    # DISALLOW MPL WARM-UP IN THE FUTURE
        Visualizer.no_instances = False
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    #                               PRIVATE
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    
    # --- UPDS ---
    def _update_scatter(self, xyz, *, colors=None):
        x, y, z = xyz
        self._scatter._offsets3d = (x, y, z)
        if colors is not None:
            self._scatter.set_facecolors(colors)

    def _update_attr_edges(self, segs, *, colors=None, opacity=None):
        self._attr.set_segments(segs)
        if colors is not None:
            self._attr.set_colors(colors)
        if opacity is not None:
            self._attr.set_alpha(opacity)

    def _update_repuls_edges(self, segs, *, colors=None, opacity=None):
        self._repuls.set_segments(segs)
        if colors is not None:
            self._repuls.set_colors(colors)
        if opacity is not None:
            self._repuls.set_alpha(opacity)

    # --- CLEARS ---

    def _clear_scatter(self):
        self._scatter._offsets3d = ([], [], [])

    def _clear_attr_edges(self):
        self._attr.set_segments(np.empty((0, 2, 3)))

    def _clear_repuls_edges(self):
        self._repuls.set_segments(np.empty((0, 2, 3)))
    
    # --- FINAL UPD ---

    def _update_canvas(self, *, pause_for: float = 0.0):
        self._fig.canvas.draw_idle()
        if pause_for > 0.0:
            plt.pause(pause_for)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    #                                PUBLIC
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

    def build_frame(
        self,
        frame_id: int,
        displayed_node_mask: Iterable | None = None,
        displayed_pairwise_attraction_mask: Iterable | None = None,
        displayed_pairwise_repulsion_mask: Iterable | None = None,
        frac_node_interactions_displayed: float = 0.01,
        node_colors: str | dict[Iterable[int], str] | None = None,
        title: str | None = None,
        padding: float = 0.1,
        spread: float = 1.0,
        show: bool = False,
        *,
        show_node_labels: bool = False,
        elev: float = 35,
        azim: float = 45,
        attractive_edge_color: str = visualizer_util.HEAT,
        repulsive_edge_color: str = visualizer_util.COLD):
        
        # PRELIMINARY:
        frame_id -= 1 # 1-base indexing

        # LOGIC:

        # threshold = np.quantile(attractive_interactions, 1 - top_percent_displayed)

