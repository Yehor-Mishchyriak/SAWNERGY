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
        edge_width: float = 1,
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
        _logger.debug("Visualizer.__init__ start | RIN_path=%s, figsize=%s, node_size=%s, edge_width=%s, depthshade=%s, antialiased=%s, init_view=(%s,%s)",
                      RIN_path, figsize, node_size, edge_width, depthshade, antialiased, init_elev, init_azim)
        if Visualizer.no_instances:
            _logger.debug("Warm-starting Matplotlib (no_instances=True).")
            visualizer_util.warm_start_matplotlib()
        else:
            _logger.debug("Skipping warm-start (no_instances=False).")
    
    # ---------- LOAD THE DATA ---------- #
        with sawnergy_util.ArrayStorage(RIN_path, mode="r") as storage:
            # FIX: correct type hints to np.ndarray (not str)
            self.COM_coords: np.ndarray      = storage.read(COM_dataset_name, slice(None))
            self.attr_energies: np.ndarray   = storage.read(attr_data_name, slice(None))
            self.repuls_energies: np.ndarray = storage.read(repuls_data_name, slice(None))
        try:
            _logger.debug("Loaded datasets | COM_coords.shape=%s, attr_energies.shape=%s, repuls_energies.shape=%s",
                          getattr(self.COM_coords, "shape", None),
                          getattr(self.attr_energies, "shape", None),
                          getattr(self.repuls_energies, "shape", None))
        except Exception:
            _logger.debug("Loaded datasets (shapes unavailable).")

        self.N = np.size(self.COM_coords[0], axis=0)
        _logger.debug("Computed N=%d", self.N)

    # - SET UP THE CANVAS AND THE AXES - #
        self._fig = plt.figure(figsize=figsize)
        self._ax  = self._fig.add_subplot(111, projection="3d")
        self._ax.set_autoscale_on(False)
        self._ax.view_init(elev=init_elev, azim=init_azim)
        self._ax.set_axis_off()
        _logger.debug("Figure and 3D axes initialized.")

    # ------ SET UP PLOT ELEMENTS ------ #
        self._scatter: PathCollection  = self._ax.scatter([], [], [], s=node_size, depthshade=depthshade, edgecolors="none")
        self._attr: Line3DCollection   = Line3DCollection(np.empty((0,2,3)), linewidths=edge_width, antialiased=antialiased)
        self._repuls: Line3DCollection = Line3DCollection(np.empty((0,2,3)), linewidths=edge_width, antialiased=antialiased)
        self._ax.add_collection3d(self._attr); self._ax.add_collection3d(self._repuls) # set pointers to the attractive and repulsive collections
        _logger.debug("Artists created | scatter(empty), attr_lines(empty), repuls_lines(empty).")

    # ---------- HELPER FIELDS --------- #
        # NOTE: 'under the hood' everything is 0-base indexed,
        # BUT, from the API point of view, the idexing is 1-base,
        # because amino acid residues are 1-base indexed.
        self._residue_norm = mpl.colors.Normalize(0,  self.N-1) # uniform coloring
        self.default_node_color = default_node_color
        _logger.debug("Helper fields set | residue_norm=[0,%d], default_node_color=%s", self.N-1, self.default_node_color)

    # DISALLOW MPL WARM-UP IN THE FUTURE
        Visualizer.no_instances = False
        _logger.debug("Visualizer.no_instances set to False.")

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    #                               PRIVATE
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    
    # --- UPDS ---
    def _update_scatter(self, xyz, *, colors=None):
        try:
            _logger.debug("_update_scatter | xyz.shape=%s, colors=%s",
                          getattr(xyz, "shape", None),
                          "provided" if colors is not None else "None")
        except Exception:
            _logger.debug("_update_scatter called (shape unavailable).")
        x, y, z = xyz.T
        self._scatter._offsets3d = (x, y, z)
        if colors is not None:
            self._scatter.set_facecolors(colors)
        _logger.debug("_update_scatter done | n_points=%s", len(x) if hasattr(x, "__len__") else "unknown")

    def _update_attr_edges(self, segs, *, colors=None, opacity=None):
        _logger.debug("_update_attr_edges | segs.shape=%s, colors=%s, opacity=%s",
                      getattr(segs, "shape", None),
                      "provided" if colors is not None else "None",
                      "array/scalar" if opacity is not None else "None")
        self._attr.set_segments(segs)
        if colors is not None and opacity is not None:
            rgba = np.array(colors, copy=True)
            if rgba.ndim == 2 and rgba.shape[1] == 4:
                rgba[:, 3] = opacity
            else:
                # map RGB to RGBA with alpha
                rgba = np.c_[rgba, np.asarray(opacity)]
            self._attr.set_colors(rgba)
        else:
            if colors is not None:
                self._attr.set_colors(colors)
            if opacity is not None:
                self._attr.set_alpha(opacity)
        _logger.debug("_update_attr_edges done.")

    def _update_repuls_edges(self, segs, *, colors=None, opacity=None):
        _logger.debug("_update_repuls_edges | segs.shape=%s, colors=%s, opacity=%s",
                      getattr(segs, "shape", None),
                      "provided" if colors is not None else "None",
                      "array/scalar" if opacity is not None else "None")
        self._repuls.set_segments(segs)
        if colors is not None and opacity is not None:
            rgba = np.array(colors, copy=True)
            if rgba.ndim == 2 and rgba.shape[1] == 4:
                rgba[:, 3] = opacity
            else:
                rgba = np.c_[rgba, np.asarray(opacity)]
            self._repuls.set_colors(rgba)
        else:
            if colors is not None:
                self._repuls.set_colors(colors)
            if opacity is not None:
                self._repuls.set_alpha(opacity)
        _logger.debug("_update_repuls_edges done.")

    # --- CLEARS ---

    def _clear_scatter(self):
        _logger.debug("_clear_scatter called.")
        self._scatter._offsets3d = ([], [], [])

    def _clear_attr_edges(self):
        _logger.debug("_clear_attr_edges called.")
        self._attr.set_segments(np.empty((0, 2, 3)))

    def _clear_repuls_edges(self):
        _logger.debug("_clear_repuls_edges called.")
        self._repuls.set_segments(np.empty((0, 2, 3)))
    
    # --- FINAL UPD ---

    def _update_canvas(self, *, pause_for: float = 0.0):
        _logger.debug("_update_canvas | pause_for=%s", pause_for)
        self._fig.canvas.draw_idle()
        if pause_for > 0.0:
            plt.pause(pause_for)

    # ADJUST THE VIEW
    def _fix_view(self, coordinates: np.ndarray, padding: float, spread: float):
        _logger.debug("_fix_view | coords.shape=%s, padding=%s, spread=%s",
                      getattr(coordinates, "shape", None), padding, spread)
        orig_min = coordinates.min(axis=0)
        orig_max = coordinates.max(axis=0)
        orig_span = np.maximum(orig_max - orig_min, 1e-12)
        xyz_min = orig_min - padding * orig_span
        xyz_max = orig_max + padding * orig_span

        self._ax.set_xlim(xyz_min[0], xyz_max[0])
        self._ax.set_ylim(xyz_min[1], xyz_max[1])
        self._ax.set_zlim(xyz_min[2], xyz_max[2])
        self._ax.set_box_aspect(np.maximum(xyz_max - xyz_min, 1e-12))
        _logger.debug("_fix_view | bounds set: x=(%s,%s), y=(%s,%s), z=(%s,%s)",
                      xyz_min[0], xyz_max[0], xyz_min[1], xyz_max[1], xyz_min[2], xyz_max[2])

        if spread != 1.0:
            center = coordinates.mean(axis=0, keepdims=True)
            coordinates = center + spread * (coordinates - center)
            _logger.debug("_fix_view | applied spread around centroid.")
        return coordinates

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    #                                PUBLIC
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

    def build_frame(
        self,
        frame_id: int,
        displayed_nodes: np.typing.ArrayLike | Literal["ALL"] | None = "ALL",
        displayed_pairwise_attraction_for_nodes: np.typing.ArrayLike | Literal["DISPLAYED_NODES"] | None = "DISPLAYED_NODES",
        displayed_pairwise_repulsion_for_nodes: np.typing.ArrayLike | Literal["DISPLAYED_NODES"] | None = "DISPLAYED_NODES",
        frac_node_interactions_displayed: float = 0.01, # 1%
        node_colors: str | dict[Iterable[int], str] | None = None,
        title: str | None = None,
        padding: float = 0.1,
        spread: float = 1.0,
        show: bool = False,
        *,
        show_node_labels: bool = False,
        node_label_size: int = 6,
        attractive_edge_cmap: str = visualizer_util.HEAT,
        repulsive_edge_cmap: str = visualizer_util.COLD):
        
        # PRELIMINARY
        _logger.debug("build_frame called | frame_id(1-based)=%s, frac_node_interactions_displayed=%s, padding=%s, spread=%s, show=%s, show_node_labels=%s",
                      frame_id, frac_node_interactions_displayed, padding, spread, show, show_node_labels)
        ALL_RESIDUES = np.arange(0, self.N, 1)
        frame_id -= 1 # 1-base indexing
        _logger.debug("build_frame | using frame_id(0-based)=%s", frame_id)

        # NODES
        if displayed_nodes is not None:
            if isinstance(displayed_nodes, str):
                if displayed_nodes == "ALL":
                    displayed_nodes = np.arange(0, self.N, 1)
                    _logger.debug("displayed_nodes='ALL' -> count=%d", displayed_nodes.size)
                else:
                    _logger.error("Invalid displayed_nodes string: %s", displayed_nodes)
                    raise ValueError(
                            "'displayed_nodes' has to be either an ArrayLike "
                            "collection of node indices, or an 'ALL' string, "
                            "or None.")
            else:
                displayed_nodes = np.asarray(displayed_nodes)-1 # 1-base indexing
                _logger.debug("displayed_nodes provided | count=%d", displayed_nodes.size)
        else:
            _logger.debug("displayed_nodes is None -> returning early.")
            return

        frame_coords = self.COM_coords[frame_id]
        nodes = frame_coords[displayed_nodes]
        _logger.debug("Selected nodes | nodes.shape=%s (before view fix)", getattr(nodes, "shape", None))

        nodes = self._fix_view(nodes, padding, spread)
        _logger.debug("Nodes after _fix_view | shape=%s", getattr(nodes, "shape", None))

        # ATTRACTIVE EDGES
        if displayed_pairwise_attraction_for_nodes is not None:
            if isinstance(displayed_pairwise_attraction_for_nodes, str):
                if displayed_pairwise_attraction_for_nodes == "DISPLAYED_NODES":
                    displayed_pairwise_attraction_for_nodes = displayed_nodes
                    _logger.debug("Attraction nodes='DISPLAYED_NODES' -> count=%d", displayed_pairwise_attraction_for_nodes.size)
                else:
                    _logger.error("Invalid attraction selector string: %s", displayed_pairwise_attraction_for_nodes)
                    raise ValueError(
                            "'displayed_pairwise_attraction_for_nodes' has to be either an ArrayLike "
                            "collection of node indices, or an 'DISPLAYED_NODES' string, "
                            "or None.")
            else:
                displayed_pairwise_attraction_for_nodes = np.asarray(displayed_pairwise_attraction_for_nodes)-1 # 1-base indexing
                _logger.debug("Attraction nodes provided | count=%d", displayed_pairwise_attraction_for_nodes.size)
            
            if np.setdiff1d(displayed_pairwise_attraction_for_nodes, displayed_nodes).size > 0:
                _logger.error("Attraction nodes not a subset of displayed_nodes.")
                raise ValueError("'displayed_pairwise_attraction_for_nodes' must be a subset of 'displayed_nodes'")
            
            attractive_edges, attractive_color_weights, attractive_opacity_weights = \
                visualizer_util.build_line_segments(
                    self.N,
                    displayed_pairwise_attraction_for_nodes,
                    nodes,
                    self.attr_energies[frame_id],
                    frac_node_interactions_displayed
                )
            _logger.debug("Attraction edges built | segs.shape=%s, color_w.shape=%s, opacity_w.shape=%s",
                          getattr(attractive_edges, "shape", None),
                          getattr(attractive_color_weights, "shape", None),
                          getattr(attractive_opacity_weights, "shape", None))

        else:
            attractive_edges = None
            _logger.debug("Attraction edges skipped (selector=None).")

        # REPULSIVE EDGES
        if displayed_pairwise_repulsion_for_nodes is not None:
            if isinstance(displayed_pairwise_repulsion_for_nodes, str):
                if displayed_pairwise_repulsion_for_nodes == "DISPLAYED_NODES":
                    displayed_pairwise_repulsion_for_nodes = displayed_nodes
                    _logger.debug("Repulsion nodes='DISPLAYED_NODES' -> count=%d", displayed_pairwise_repulsion_for_nodes.size)
                else:
                    _logger.error("Invalid repulsion selector string: %s", displayed_pairwise_repulsion_for_nodes)
                    raise ValueError(
                            "'displayed_pairwise_repulsion_for_nodes' has to be either an ArrayLike "
                            "collection of node indices, or an 'DISPLAYED_NODES' string, "
                            "or None.")
            else:
                displayed_pairwise_repulsion_for_nodes = np.asarray(displayed_pairwise_repulsion_for_nodes)-1 # 1-base indexing
                _logger.debug("Repulsion nodes provided | count=%d", displayed_pairwise_repulsion_for_nodes.size)
            
            if np.setdiff1d(displayed_pairwise_repulsion_for_nodes, displayed_nodes).size > 0:
                _logger.error("Repulsion nodes not a subset of displayed_nodes.")
                raise ValueError("'displayed_pairwise_repulsion_for_nodes' must be a subset of 'displayed_nodes'")
            
            repulsive_edges, repulsive_color_weights, repulsive_opacity_weights = \
                visualizer_util.build_line_segments(
                    self.N,
                    displayed_pairwise_repulsion_for_nodes,
                    nodes,
                    self.repuls_energies[frame_id],
                    frac_node_interactions_displayed
                )
            _logger.debug("Repulsion edges built | segs.shape=%s, color_w.shape=%s, opacity_w.shape=%s",
                          getattr(repulsive_edges, "shape", None),
                          getattr(repulsive_color_weights, "shape", None),
                          getattr(repulsive_opacity_weights, "shape", None))

        else:
            repulsive_edges = None
            _logger.debug("Repulsion edges skipped (selector=None).")

        # COLOR THE DATA POINTS
        if isinstance(node_colors, str):
            node_cmap = plt.get_cmap(node_colors)
            idx0 = np.asarray(displayed_nodes, dtype=int)
            color_array = node_cmap(self._residue_norm(idx0))
            _logger.debug("Node colors via colormap '%s' | count=%d", node_colors, idx0.size)
        else:
            color_array_full = visualizer_util.map_groups_to_colors(
                N=self.N,
                groups= {} if node_colors is None else node_colors,
                default_color=self.default_node_color,
                one_based=True
            )
            color_array = np.asarray(color_array_full)[displayed_nodes]
            _logger.debug("Node colors via groups/default | count=%d", color_array.shape[0])

        # UPDATE CANVAS
        self._update_scatter(nodes, colors=color_array)
        
        if attractive_edges is not None:
            attractive_cmap = plt.get_cmap(attractive_edge_cmap)
            attr_rgba = attractive_cmap(attractive_color_weights)         # (E,4)
            attr_rgba[:, 3] = repulsive_opacity_weights if False else attractive_opacity_weights 
            self._update_attr_edges(attractive_edges,
                                    colors=attr_rgba,
                                    opacity=None)
            _logger.debug("Attraction edges updated on canvas.")
        
        if repulsive_edges is not None:
            repulsive_cmap = plt.get_cmap(repulsive_edge_cmap)
            rep_rgba = repulsive_cmap(repulsive_color_weights)            # (E,4)
            rep_rgba[:, 3] = repulsive_opacity_weights
            self._update_repuls_edges(repulsive_edges,
                                      colors=rep_rgba,
                                      opacity=None)
            _logger.debug("Repulsion edges updated on canvas.")
  
        # EXTRAS
        if title:
            self._fig.suptitle(title)
            _logger.debug("Title set: %s", title)

        if show_node_labels:
            labs = (np.asarray(displayed_nodes, dtype=int) + 1) # one-based labels
            _logger.debug("Adding node labels | count=%d, fontsize=%d", labs.size, node_label_size)
            for (x, y, z), lab in zip(nodes, labs):
                self._ax.text(float(x), float(y), float(z), str(lab),
                              fontsize=node_label_size, color="k")
        
        if show:
            # auto-block in scripts; non-block in notebooks/interactive
            try:
                get_ipython  # type: ignore
                in_ipy = True
            except NameError:
                in_ipy = False

            _logger.debug("Showing figure | in_ipy=%s, interactive=%s", in_ipy, plt.isinteractive())

            if in_ipy or plt.isinteractive():
                plt.show(block=False)
                plt.pause(0.05)
            else:
                plt.show()
        _logger.debug("build_frame completed.")


__all__ = [
    "Visualizer"
]

if __name__ == "__main__":
    pass
