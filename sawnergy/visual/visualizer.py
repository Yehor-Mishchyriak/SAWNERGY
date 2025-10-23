from __future__ import annotations

# third-pary
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import PathCollection
# built-in
from pathlib import Path
from typing import Iterable, Literal
import logging
# local
from . import visualizer_util
from .. import sawnergy_util

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__)

# *----------------------------------------------------*
#                        CLASSES
# *----------------------------------------------------*

class Visualizer:
    """3D network/trajectory visualizer.

    This class renders nodes (scatter) and pairwise interactions (line segments)
    for frames of a trajectory stored in an ArrayStorage-backed file (e.g., Zarr
    in a ZIP). It supports showing only a subset of nodes, coloring nodes by
    groups or a colormap, drawing attractive/repulsive edges by weight quantiles,
    and animating the full trajectory.

    Backend & GUI behavior:
        - The Matplotlib backend is chosen in __init__ via
          visualizer_util.ensure_backend(show), *before* importing pyplot.
          If `show=True` but no GUI/display is available (e.g., headless Linux),
          the backend is switched to 'Agg' and a warning is emitted. In this mode
          figures render off-screen; use savefig() instead of interactive windows.
        - pyplot is imported lazily inside __init__ after backend selection and
          stored as `self._plt` to keep backend control deterministic.

    Attributes:
        no_instances: Class-level flag to warm-start Matplotlib only once.
        COM_coords: Trajectory coordinates, shape (T, N, 3).
        attr_energies: Attractive weights (shape (T, N, N)) or None if absent.
        repuls_energies: Repulsive weights (shape (T, N, N)) or None if absent.
        N: Number of nodes (int).
        _fig: Matplotlib Figure.
        _ax: 3D Axes.
        _scatter: PathCollection for node markers.
        _attr: Line3DCollection for attractive edges.
        _repuls: Line3DCollection for repulsive edges.
        _residue_norm: Normalizer mapping [0, N-1] to [0, 1] for colormaps.
        default_node_color: Hex color string used when no group color is set.
    """

    no_instances: bool = True

    def __init__(
        self,
        RIN_path: str | Path,
        figsize: tuple[int, int] = (9, 7),
        node_size: int = 120,
        edge_width: float = 1.25,
        default_node_color: str = visualizer_util.GRAY,
        depthshade: bool = False,
        antialiased: bool = False,
        init_elev: float = 35,
        init_azim: float = 45,
        *,
        show: bool = False
    ) -> None:
        """Initialize the visualizer and load datasets.

        Args:
            RIN_path: Path to the archive or store containing datasets.
            figsize: Figure size (inches) for the Matplotlib window.
            node_size: Marker area for nodes (passed to `Axes3D.scatter`).
            edge_width: Line width for edge collections.
            default_node_color: Hex color used for nodes not in any group.
            depthshade: Whether to apply depth shading to scatter points.
            antialiased: Whether to antialias line collections.
            init_elev: Initial elevation angle (degrees) for 3D view.
            init_azim: Initial azimuth angle (degrees) for 3D view.
            show: Hint about intended usage. If True and a GUI/display is available,
                interactive windows can be shown later (e.g., via `self._plt.show()`).
                If True but no GUI/display is available, the backend is switched to
                'Agg' (off-screen) and a warning is issued. This flag does not itself
                call `show()`; it only influences backend selection.

        Data discovery:
            Dataset names are auto-resolved from storage attrs:
            'com_name', 'attractive_energies_name', 'repulsive_energies_name'.
            Any missing channel remains disabled (None) but visualization still works
            just without edges of a specific missing type.

        Side Effects:
            - Selects a Matplotlib backend before importing pyplot; may fall back
                to 'Agg' in headless environments when `show=True`.
            - Optionally warms up Matplotlib once per process (first instance only).
            - Opens and reads required datasets from storage.
            - Creates a figure, 3D axes, and empty artists for later updates.
        """
        # choose GUI backend before importing pyplot
        visualizer_util.ensure_backend(show)
        import matplotlib.pyplot as plt
        self._plt = plt
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
            com_name = storage.get_attr("com_name")
            attr_energies_name = storage.get_attr("attractive_energies_name")
            repuls_energies_name = storage.get_attr("repulsive_energies_name")
            self.COM_coords: np.ndarray      = storage.read(com_name, slice(None))
            self.attr_energies: np.ndarray   = storage.read(attr_energies_name, slice(None)) if attr_energies_name is not None else None
            self.repuls_energies: np.ndarray = storage.read(repuls_energies_name, slice(None)) if repuls_energies_name is not None else None
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
        self._fig = plt.figure(figsize=figsize, num="SAWNERGY")
        self._ax  = self._fig.add_subplot(111, projection="3d")
        self._fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self._fig.patch.set_facecolor("#999999")
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
        # BUT, from the API point of view, the indexing is 1-base,
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
        """Update scatter artist with new positions (and optional colors).

        Args:
            xyz: Array-like of shape (N_visible, 3) containing node positions
                for the *currently displayed* nodes.
            colors: Optional array of RGBA colors (len=N_visible) or a single
                color broadcastable by Matplotlib.

        Returns:
            None
        """
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
        """Update attractive edge collection.

        Args:
            segs: Array of shape (E, 2, 3) with edge endpoints.
            colors: Optional array of RGB/RGBA per-edge colors or a single color.
            opacity: Optional array or scalar alpha(s). If both `colors` and
                `opacity` are provided, alpha will be fused into the RGBA.

        Returns:
            None
        """
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
        """Update repulsive edge collection.

        Args:
            segs: Array of shape (E, 2, 3) with edge endpoints.
            colors: Optional array of RGB/RGBA per-edge colors or a single color.
            opacity: Optional array or scalar alpha(s). If both `colors` and
                `opacity` are provided, alpha will be fused into the RGBA.

        Returns:
            None
        """
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
        """Clear node positions from the scatter artist.

        Returns:
            None
        """
        _logger.debug("_clear_scatter called.")
        self._scatter._offsets3d = ([], [], [])

    def _clear_attr_edges(self):
        """Clear all attractive edges from the collection.

        Returns:
            None
        """
        _logger.debug("_clear_attr_edges called.")
        self._attr.set_segments(np.empty((0, 2, 3)))

    def _clear_repuls_edges(self):
        """Clear all repulsive edges from the collection.

        Returns:
            None
        """
        _logger.debug("_clear_repuls_edges called.")
        self._repuls.set_segments(np.empty((0, 2, 3)))
    
    # --- FINAL UPD ---

    def _update_canvas(self, *, pause_for: float = 0.0):
        """Request a canvas redraw and optionally pause.

        Args:
            pause_for: If > 0, calls `plt.pause(pause_for)` to advance GUI
                event loops and create a visible delay (useful in animations).

        Returns:
            None
        """
        _logger.debug("_update_canvas | pause_for=%s", pause_for)
        self._fig.canvas.draw_idle()
        if pause_for > 0.0:
            self._plt.pause(pause_for)

    # ADJUST THE VIEW
    def _fix_view(self, coordinates: np.ndarray, padding: float, spread: float):
        """Adjust axes limits/box aspect and apply optional spatial spreading.

        Computes a bounding box around provided coordinates, expands it by
        ``padding`` (relative to span), sets axes limits and box aspect, and
        optionally spreads points around their centroid by factor ``spread``.

        Args:
            coordinates: Array of shape (M, 3) for currently displayed nodes.
            padding: Fraction of the original span added to min/max on each axis.
            spread: If != 1.0, multiply deviations from centroid by this factor.

        Returns:
            np.ndarray: Possibly modified copy of `coordinates` (same shape) if
            `spread` was applied; otherwise the input array is returned.
        """
        _logger.debug("_fix_view | coords.shape=%s, padding=%s, spread=%s",
                        getattr(coordinates, "shape", None), padding, spread)

        # Apply spread first so limits reflect the final positions
        if spread != 1.0:
            center = coordinates.mean(axis=0, keepdims=True)
            coordinates = center + spread * (coordinates - center)
            _logger.debug("_fix_view | applied spread around centroid.")

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
        global_interactions_frac: bool = True,
        global_opacity: bool = True,
        global_color_saturation: bool = True,
        node_colors: str | tuple[tuple[Iterable[int], str]] | None = None,
        title: str | None = None,
        padding: float = 0.1,
        spread: float = 1.0,
        show: bool = False,
        *,
        show_node_labels: bool = False,
        node_label_size: int = 6,
        attractive_edge_cmap: str = visualizer_util.HEAT,
        repulsive_edge_cmap: str = visualizer_util.COLD):
        """Render a single frame into existing artists.

        This updates node positions/colors and draws attractive/repulsive
        edges chosen by a quantile threshold on weights. Indices passed from
        the public API are interpreted as 1-based and converted to 0-based
        internally.

        Args:
            frame_id: 1-based frame index to render.
            displayed_nodes: Iterable of node indices to show (1-based),
                "ALL" for all nodes, or None to return early.
            displayed_pairwise_attraction_for_nodes: Iterable of node indices
                (1-based) or the literal "DISPLAYED_NODES" to restrict
                candidate attractive edges to those whose endpoints are both
                in this set.
            displayed_pairwise_repulsion_for_nodes: Same contract as
                `displayed_pairwise_attraction_for_nodes` but for repulsive edges.
            frac_node_interactions_displayed: Fraction of heaviest edges to keep
                (approximate top-`frac`) after endpoint filtering.
            global_interactions_frac: If True, the threshold uses all
                upper-triangle weights; otherwise only candidate edges.
            global_opacity: If True, opacity uses global row-wise normalization;
                otherwise, normalization uses only kept edges (others set to 0).
            global_color_saturation: If True, color uses global absolute
                normalization; otherwise, normalization uses only kept edges.
            node_colors: Either a Matplotlib colormap name (str) or a tuple of
                (indices, hex_color) group pairs for per-node colors.
            title: Optional title displayed in axis coordinates.
            padding: Fractional padding around the displayed nodes' bounds.
            spread: Spatial spread multiplier applied about centroid (displayed
                nodes only).
            show: If True, request a window display:
                - In IPython/interactive mode, uses non-blocking `show(block=False)` and
                a short `pause()` to flush the GUI loop.
                - In non-interactive scripts, uses blocking `show()` at the end
                of the draw step.
                - If the backend is 'Agg' (headless fallback), this is a no-op for
                windows; use `self._plt.savefig(...)` to persist images.

        Returns:
            None

        Raises:
            ValueError: If any requested attraction/repulsion node set is not a
                subset of `displayed_nodes`, or invalid sentinel strings are used.
        """
        # PRELIMINARY
        _logger.debug("build_frame called | frame_id(1-based)=%s, frac_node_interactions_displayed=%s, padding=%s, spread=%s, show=%s, show_node_labels=%s",
                      frame_id, frac_node_interactions_displayed, padding, spread, show, show_node_labels)
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
        coords_for_edges = frame_coords.copy()
        coords_for_edges[displayed_nodes] = nodes

        # ATTRACTIVE EDGES
        if displayed_pairwise_attraction_for_nodes is not None:
            if self.attr_energies is None:
                _logger.warning("Attractive dataset unavailable; skipping attractive edges.")
                attractive_edges = None
            else:
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
                        coords_for_edges,
                        self.attr_energies[frame_id],
                        frac_node_interactions_displayed,
                        global_weights_frac=global_interactions_frac,
                        global_opacity=global_opacity,
                        global_color_saturation=global_color_saturation
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
            if self.repuls_energies is None:
                _logger.warning("Repulsive dataset unavailable; skipping repulsive edges.")
                repulsive_edges = None
            else:
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
                        coords_for_edges,
                        self.repuls_energies[frame_id],
                        frac_node_interactions_displayed,
                        global_weights_frac=global_interactions_frac,
                        global_opacity=global_opacity,
                        global_color_saturation=global_color_saturation
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
            node_cmap = self._plt.get_cmap(node_colors)
            idx0 = np.asarray(displayed_nodes, dtype=int)
            color_array = node_cmap(self._residue_norm(idx0))
            _logger.debug("Node colors via colormap '%s' | count=%d", node_colors, idx0.size)
        else:
            color_array_full = visualizer_util.map_groups_to_colors(
                N=self.N,
                groups=node_colors,
                default_color=self.default_node_color,
                one_based=True
            )
            color_array = np.asarray(color_array_full)[displayed_nodes]
            _logger.debug("Node colors via groups/default | count=%d", color_array.shape[0])

        # UPDATE CANVAS
        self._update_scatter(nodes, colors=color_array)
        
        if attractive_edges is not None:
            attractive_cmap = self._plt.get_cmap(attractive_edge_cmap)
            attr_rgba = attractive_cmap(attractive_color_weights)         # (E,4)
            attr_rgba[:, 3] = attractive_opacity_weights
            self._update_attr_edges(attractive_edges,
                                    colors=attr_rgba,
                                    opacity=None)
            _logger.debug("Attraction edges updated on canvas.")
        
        if repulsive_edges is not None:
            repulsive_cmap = self._plt.get_cmap(repulsive_edge_cmap)
            rep_rgba = repulsive_cmap(repulsive_color_weights)            # (E,4)
            rep_rgba[:, 3] = repulsive_opacity_weights
            self._update_repuls_edges(repulsive_edges,
                                      colors=rep_rgba,
                                      opacity=None)
            _logger.debug("Repulsion edges updated on canvas.")
  
        # EXTRAS
        if title:
            self._ax.text2D(0.5, 0.99, title, transform=self._ax.transAxes,
                            ha="center", va="top")
            _logger.debug("Title set: %s", title)

        if show_node_labels:
            labs = (np.asarray(displayed_nodes, dtype=int) + 1) # one-based labels
            _logger.debug("Adding node labels | count=%d, fontsize=%d", labs.size, node_label_size)
            for (x, y, z), lab in zip(nodes, labs):
                self._ax.text(float(x)+.3, float(y)+.3, float(z)+1.3, str(lab),
                              fontsize=node_label_size, color="k")
        
        if show:
            # auto-block in scripts; non-block in notebooks/interactive
            try:
                get_ipython  # type: ignore
                in_ipy = True
            except NameError:
                in_ipy = False

            _logger.debug("Showing figure | in_ipy=%s, interactive=%s", in_ipy, self._plt.isinteractive())

            if in_ipy or self._plt.isinteractive():
                self._plt.show(block=False)
                self._plt.pause(0.05)
            else:
                self._plt.show()
        _logger.debug("build_frame completed.")

    def animate_trajectory(
        self,
        start: int = 1,
        stop: int | None = None,
        step: int = 1,
        interval_ms: int = 50,
        loop: bool = False,
        **build_kwargs,
    ):
        """Play frames as an animation by reusing existing artists.

        Iterates frames from `start` to `stop` (inclusive, stepping by `step`)
        and calls `build_frame` for each, pausing `interval_ms` between
        updates. If `loop=True`, the sequence repeats until the figure is
        closed or the user interrupts.

        Args:
            start: 1-based starting frame index.
            stop: 1-based ending frame index (inclusive). Defaults to the last
                available frame if None.
            step: Step size between frames. Negative values play backwards.
            interval_ms: Pause between frames in milliseconds.
            loop: If True, repeat the frame sequence indefinitely (until the
                figure is closed or interrupted).
            **build_kwargs: Additional keyword arguments forwarded to
                `build_frame` (e.g., `displayed_nodes`, `padding`, `spread`,
                `node_colors`, etc.). `show=False` is enforced internally.

        Returns:
            None

        Raises:
            ValueError: If `step` is zero.
            
        Notes:
            - Internally enforces `build_kwargs["show"] = False` during iteration to
            avoid blocking; a final `self._plt.show()` is issued at the end of a
            single pass.
            - In headless mode (backend 'Agg'), no GUI window appears; use
            `self._plt.savefig(...)` or a writer to export frames.
        """
        _logger.debug(
            "animate_trajectory | start=%s, stop=%s, step=%s, interval_ms=%s, loop=%s",
            start, stop, step, interval_ms, loop
        )

        # default to all frames
        T = int(self.COM_coords.shape[0])
        if stop is None:
            stop = T

        if step == 0:
            raise ValueError("step must be non-zero")

        # build the list of frame ids
        frames = list(range(start, stop + (1 if step > 0 else -1), step)) # allow for backward play
        if not frames:
            _logger.debug("animate_trajectory | empty frame list -> return")
            return

        build_kwargs["show"] = False

        try:
            if loop:
                _logger.debug("animate_trajectory | entering repeat loop until window closed.")
                while self._plt.fignum_exists(self._fig.number):
                    for fid in frames:
                        if not self._plt.fignum_exists(self._fig.number):
                            break
                        self.build_frame(fid, **build_kwargs)
                        self._update_canvas(pause_for=interval_ms / 1000.0)
            else:
                _logger.debug("animate_trajectory | single pass over frames.")
                for fid in frames:
                    self.build_frame(fid, **build_kwargs)
                    self._update_canvas(pause_for=interval_ms / 1000.0)
                # one final show so the window stays up when the loop ends
                self._plt.show()
        except KeyboardInterrupt:
            _logger.debug("animate_trajectory | interrupted by user (KeyboardInterrupt).")


__all__ = [
    "Visualizer"
]

if __name__ == "__main__":
    pass
