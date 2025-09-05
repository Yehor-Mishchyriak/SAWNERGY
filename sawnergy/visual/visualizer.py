from __future__ import annotations

# third-pary
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import PathCollection
# built-in
from pathlib import Path
from typing import Iterable, Literal
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
        init_elev: float = 35,
        init_azim: float = 45,
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

        self.N = np.size(self.COM_coords[0], axis=0)

    # - SET UP THE CANVAS AND THE AXES - #
        self._fig = plt.figure(figsize=figsize)
        self._ax  = self._fig.add_subplot(111, projection="3d")
        self._ax.set_autoscale_on(False)
        self._ax.view_init(elev=init_elev, azim=init_azim)
        self._ax.set_axis_off()
    
    # ------ SET UP PLOT ELEMENTS ------ #
        self._scatter: PathCollection  = self._ax.scatter([], [], [], s=node_size, depthshade=depthshade, edgecolors="none")
        self._attr: Line3DCollection   = Line3DCollection(np.empty((0,2,3)), linewidths=edge_width, antialiased=antialiased)
        self._repuls: Line3DCollection = Line3DCollection(np.empty((0,2,3)), linewidths=edge_width, antialiased=antialiased)
        self._ax.add_collection3d(self._attr); self._ax.add_collection3d(self._repuls) # set pointers to the attractive and repulsive collections

    # ---------- HELPER FIELDS --------- #
        # NOTE: 'under the hood' everything is 0-base indexed,
        # BUT, from the API point of view, the idexing is 1-base,
        # because amino acid residues are 1-base indexed.
        self._residue_norm = mpl.colors.Normalize(0,  self.N-1) # uniform coloring
        self.default_node_color = default_node_color

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

    # ADJUST THE VIEW
    def _fix_view(self, coordinates: np.ndarray, padding: float, spread: float):
        orig_min = coordinates.min(axis=0)
        orig_max = coordinates.max(axis=0)
        orig_span = np.maximum(orig_max - orig_min, 1e-12)
        xyz_min = orig_min - padding * orig_span
        xyz_max = orig_max + padding * orig_span

        self._ax.set_xlim(xyz_min[0], xyz_max[0])
        self._ax.set_ylim(xyz_min[1], xyz_max[1])
        self._ax.set_zlim(xyz_min[2], xyz_max[2])
        self._ax.set_box_aspect(np.maximum(xyz_max - xyz_min, 1e-12))

        if spread != 1.0:
            center = coordinates.mean(axis=0, keepdims=True)
            coordinates = center + spread * (coordinates - center)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    #                                PUBLIC
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

    def build_frame(
        self,
        frame_id: int,
        displayed_nodes: np.typing.ArrayLike | Literal["ALL"] | None = "ALL",
        dispalyed_pairwise_attraction_for_nodes: np.typing.ArrayLike | Literal["ALL"] | None = "ALL",
        dispalyed_pairwise_repulsion_for_nodes: np.typing.ArrayLike | Literal["ALL"] | None = "ALL",
        frac_node_interactions_displayed: float = 0.01, # 1%
        node_colors: str | dict[Iterable[int], str] | None = None,
        title: str | None = None,
        padding: float = 0.1,
        spread: float = 1.0,
        show: bool = False,
        *,
        show_node_labels: bool = False,
        attractive_edge_cmap: str = visualizer_util.HEAT,
        repulsive_edge_cmap: str = visualizer_util.COLD):
        
        # PRELIMINARY
        ALL_RESIDUES = np.arange(0, self.N, 1)
        frame_id -= 1 # 1-base indexing

        # NODES
        if displayed_nodes is not None:
            if isinstance(displayed_nodes, str):
                if displayed_nodes == "ALL":
                    displayed_nodes = ALL_RESIDUES
                else:
                    raise ValueError(
                            "'displayed_nodes' has to be either an ArrayLike "
                            "collection of node indices, or an 'ALL' string, "
                            "or None.")
            else:
                displayed_nodes = np.asarray(displayed_nodes)-1 # 1-base indexing
        else:
            return

        nodes = self.COM_coords[displayed_nodes]

        self._fix_view(nodes, padding, spread)

        # ATTRACTIVE EDGES
        if dispalyed_pairwise_attraction_for_nodes is not None:
            if isinstance(dispalyed_pairwise_attraction_for_nodes, str):
                if dispalyed_pairwise_attraction_for_nodes == "ALL":
                    dispalyed_pairwise_attraction_for_nodes = ALL_RESIDUES
                else:
                    raise ValueError(
                            "'dispalyed_pairwise_attraction_for_nodes' has to be either an ArrayLike "
                            "collection of node indices, or an 'ALL' string, "
                            "or None.")
            else:
                dispalyed_pairwise_attraction_for_nodes = np.asarray(dispalyed_pairwise_attraction_for_nodes)-1 # 1-base indexing
            
            if np.setdiff1d(dispalyed_pairwise_attraction_for_nodes, displayed_nodes).size > 0:
                raise ValueError("'dispalyed_pairwise_attraction_for_nodes' must be a subset of 'displayed_nodes'")
            
            attractive_edges, attractive_color_weights, attractive_opacity_weights = \
                visualizer_util.build_line_segments(
                    self.N,
                    dispalyed_pairwise_attraction_for_nodes,
                    self.COM_coords[frame_id],
                    self.attr_energies[frame_id],
                    frac_node_interactions_displayed
                )

        else:
            attractive_edges = None

        # REPULSIVE EDGES
        if dispalyed_pairwise_repulsion_for_nodes is not None:
            if isinstance(dispalyed_pairwise_repulsion_for_nodes, str):
                if dispalyed_pairwise_repulsion_for_nodes == "ALL":
                    dispalyed_pairwise_repulsion_for_nodes = ALL_RESIDUES
                else:
                    raise ValueError(
                            "'dispalyed_pairwise_repulsion_for_nodes' has to be either an ArrayLike "
                            "collection of node indices, or an 'ALL' string, "
                            "or None.")
            else:
                dispalyed_pairwise_repulsion_for_nodes = np.asarray(dispalyed_pairwise_repulsion_for_nodes)-1 # 1-base indexing
            
            if np.setdiff1d(dispalyed_pairwise_repulsion_for_nodes, displayed_nodes).size > 0:
                raise ValueError("'dispalyed_pairwise_repulsion_for_nodes' must be a subset of 'displayed_nodes'")
            
            repulsive_edges, repulsive_color_weights, repulsive_opacity_weights = \
                visualizer_util.build_line_segments(
                    self.N,
                    dispalyed_pairwise_repulsion_for_nodes,
                    self.COM_coords[frame_id],
                    self.repuls_energies[frame_id],
                    frac_node_interactions_displayed
                )

        else:
            repulsive_edges = None

        # COLOR THE DATA POINTS
        if isinstance(node_colors, str):
            node_cmap = plt.get_cmap(node_colors)
            node_colors = mpl.cm.ScalarMappable(
                norm=self._residue_norm,
                cmap=node_cmap
            )
            cbar = self._fig.colorbar(node_colors, ax=self._ax, fraction=0.025, pad=0.04)
            cbar.set_label(f"Residue index 1 → {self.N}")
            cbar.set_ticks([1, self.N])
        else:
            node_colors = visualizer_util.map_groups_to_colors(
                N=self.N,
                groups=node_colors,
                default_color=self.default_node_color,
                one_based=True
            )

        # UPDATE CANVAS
        self._update_scatter(nodes, colors=node_colors)
        
        if attractive_edges is not None:
            attractive_cmap = plt.get_cmap(attractive_edge_cmap)
            self._update_attr_edges(attractive_edges,
                                colors=attractive_cmap(attractive_color_weights),
                                opacity=attractive_opacity_weights)
        
        if repulsive_edges is not None:
            repulsive_cmap = plt.get_cmap(repulsive_edge_cmap)
            self._update_repuls_edges(repulsive_edges,
                                  colors=repulsive_cmap(repulsive_color_weights),
                                  opacity=repulsive_opacity_weights)
  
        # EXTRAS
        if title:
            self._fig.suptitle(title)
        
