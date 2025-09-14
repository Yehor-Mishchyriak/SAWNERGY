# third-pary
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
# built-in
from typing import Iterable
import logging

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__)

# DISCRETE
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

# CONTINUOUS SPECTRUM
HEAT = "autumn"
COLD = "winter"

# *----------------------------------------------------*
#                       FUNCTIONS
# *----------------------------------------------------*

# -=-=-=-=-=-=-=-=-=-=-=- #
#       CONVENIENCE     
# -=-=-=-=-=-=-=-=-=-=-=- #

def ensure_backend(show: bool) -> None:
    """
    If the user asked to show a window but no GUI is available, switch to Agg and warn.
    Must be called *before* importing matplotlib.pyplot.
    """
    import os, sys, matplotlib, warnings, logging
    headless = (
        sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
    )
    if show and headless:
        matplotlib.use("Agg", force=True)
        warnings.warn(
            "No GUI/display detected. Falling back to non-interactive 'Agg' backend. "
            "Figures will be saved to files instead of shown."
        )
        logging.getLogger(__name__).warning(
            "Headless environment detected; switched Matplotlib backend to 'Agg'."
        )

def warm_start_matplotlib() -> None:
    """Prime Matplotlib caches and the 3D pipeline.

    This function performs a lightweight warm-up to avoid the first-draw stall
    often seen in Matplotlib, especially when using 3D axes and colorbars.
    It preloads the font manager and triggers a minimal 3D render.

    Side Effects:
        Initializes Matplotlib's font cache and issues a tiny 3D draw with a
        colorbar, then closes the temporary figure.

    Raises:
        This function intentionally swallows all exceptions and logs them at
        DEBUG level; nothing is raised to the caller.
    """
    _logger.debug("warm_start_matplotlib: starting.")
    try:
        from matplotlib import font_manager
        _ = font_manager.findSystemFonts()
        _ = font_manager.FontManager()
        _logger.debug("warm_start_matplotlib: font manager primed.")
    except Exception as e:
        _logger.debug("warm_start_matplotlib: font warmup failed: %s", e)
    try:
        # tiny 3D figure + colormap + initial render
        f = plt.figure(figsize=(1, 1))
        ax = f.add_subplot(111, projection="3d")
        ax.plot([0, 1], [0, 1], [0, 1])
        f.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=ax, fraction=0.2, pad=0.04)
        f.canvas.draw_idle()
        plt.pause(0.01)
        plt.close(f)
        _logger.debug("warm_start_matplotlib: 3D pipeline primed.")
    except Exception as e:
        _logger.debug("warm_start_matplotlib: 3D warmup failed: %s", e)

def map_groups_to_colors(N: int,
                        groups: tuple[tuple[Iterable[int], str]] | None,
                        default_color: str,
                        one_based: bool = True):
    """Map index groups to RGBA colors.

    Builds an RGBA color array of length ``N`` initialized to ``default_color``,
    then overwrites entries specified by the provided groups.

    Args:
        N: Total number of items (length of the output color array).
        groups: An optional tuple of ``(indices, color_hex)`` pairs, where
            ``indices`` is any iterable of int indices and ``color_hex`` is a
            Matplotlib-parsable color (e.g., ``"#EF4444"``). If ``None``, all
            entries are set to ``default_color``.
        default_color: Fallback color used for all indices not covered by
            ``groups``.
        one_based: If ``True``, the indices in each group are interpreted as
            1-based and will be converted internally to 0-based. If ``False``,
            indices are used as-is.

    Returns:
        list[tuple[float, float, float, float]]: A list of RGBA tuples of length
        ``N`` suitable for Matplotlib facecolors.

    Raises:
        IndexError: If any provided index is out of ``[0, N-1]`` after
            converting from 1-based (when ``one_based=True``).

    Notes:
        - No deduplication is performed across groups; later groups overwrite
          earlier ones for the same index.
    """
    _logger.debug("map_groups_to_colors: N=%s, groups=%s, default_color=%s, one_based=%s",
                  N, None if groups is None else len(groups), default_color, one_based)
    base = mcolors.to_rgba(default_color)
    colors = [base for _ in range(N)]
    if groups is not None:
        for indices, hex_color in groups:
            col = mcolors.to_rgba(hex_color)
            for idx in indices:
                i = (idx - 1) if one_based else idx
                if not (0 <= i < N):
                    _logger.error("map_groups_to_colors: index %s out of range for N=%s", idx, N)
                    raise IndexError(f"Index {idx} out of range for N={N}")
                colors[i] = col
        _logger.debug("map_groups_to_colors: completed.")
    return colors

# -=-=-=-=-=-=-=-=-=-=-=- #
#    SCENE CONSTRUCTION     
# -=-=-=-=-=-=-=-=-=-=-=- #

def absolute_quantile(N: int, weights: np.ndarray, frac: float) -> float:
    """Compute a global upper-triangle weight quantile threshold.

    Extracts the upper triangular (k=1) entries of the ``weights`` matrix for a
    graph of size ``N`` and returns the quantile corresponding to
    ``1.0 - frac``. For example, ``frac=0.25`` yields the 75th percentile.

    Args:
        N: Number of nodes (size of the square ``weights`` matrix).
        weights: 2D array of shape ``(N, N)`` containing edge weights.
        frac: Fraction in ``[0, 1]`` representing the *top* share of edges to
            keep (e.g., 0.01 means top 1%). Internally converts to the
            ``1.0 - frac`` quantile.

    Returns:
        float: The threshold value such that edges >= threshold correspond
        approximately to the top ``frac`` of the (upper-triangle) weights.

    Raises:
        ValueError: If ``weights`` has incompatible shape with ``N`` (not
            explicitly validated here, but downstream NumPy may raise).
    """
    _logger.debug("absolute_quantile: N=%s, weights.shape=%s, frac=%s",
                  N, getattr(weights, "shape", None), frac)
    r, c = np.triu_indices(N, k=1)
    vals = weights[r, c]
    if vals.size == 0:
        _logger.debug("absolute_quantile: no upper-tri edges; returning 0.0")
        return 0.0
    q = float(np.quantile(vals, 1.0 - frac))
    _logger.debug("absolute_quantile: computed threshold=%s over %d edges", q, vals.size)
    return q

def row_wise_norm(weights: np.ndarray) -> np.ndarray:
    """Normalize an adjacency/weight matrix row-wise.

    Each row is divided by its row sum. Rows with zero sum become all zeros
    (no NaNs/inf).

    Args:
        weights: 2D array, typically ``(N, N)`` adjacency/weight matrix.

    Returns:
        np.ndarray: Same shape as ``weights`` with each row summing to ~1, or
        all zeros for zero-sum rows.
    """
    _logger.debug("row_wise_norm: weights.shape=%s", getattr(weights, "shape", None))
    sums = np.sum(weights, axis=1, keepdims=True)
    out = np.zeros_like(weights, dtype=float)
    np.divide(weights, sums, out=out, where=(sums != 0))
    try:
        _logger.debug("row_wise_norm: row_sums[min=%.6g, max=%.6g]", float(sums.min()), float(sums.max()))
    except Exception:
        _logger.debug("row_wise_norm: row_sums stats unavailable.")
    return out

def absolute_norm(weights: np.ndarray) -> np.ndarray:
    """Normalize an array by its total sum.

    Divides all entries by the total sum. If the total is zero/non-finite,
    returns an all-zeros array (no NaNs/inf).

    Args:
        weights: Array (any shape). Often an ``(N, N)`` matrix of edge weights.

    Returns:
        np.ndarray: Same shape as ``weights`` whose values sum to ~1, or zeros
        if the total is zero/non-finite.
    """
    _logger.debug("absolute_norm: weights.shape=%s", getattr(weights, "shape", None))
    total = np.sum(weights)
    if not np.isfinite(total) or total == 0:
        _logger.debug("absolute_norm: total_sum is zero/non-finite; returning zeros")
        return np.zeros_like(weights, dtype=float)
    out = weights / total
    try:
        _logger.debug("absolute_norm: total_sum=%.6g", float(total))
    except Exception:
        _logger.debug("absolute_norm: total_sum unavailable.")
    return out

def build_line_segments(
    N: int,
    include: np.ndarray, 
    coords: np.ndarray,
    weights: np.ndarray,
    top_frac_weights_displayed: float,
    *,
    global_weights_frac: bool = True,
    global_opacity: bool = True,
    global_color_saturation: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct 3D line segments and per-edge style weights.

    Builds edge segments between selected node pairs and returns arrays used for
    color and opacity weighting. The selection is performed by:
    1) keeping only edges whose both endpoints are in ``include``, then
    2) thresholding by a quantile on weights (globally or only among candidate
       edges), keeping approximately the top ``top_frac_weights_displayed``.

    Args:
        N: Total number of nodes (size of the weight matrix).
        include: 1D array of node indices to consider (0-based).
        coords: Array of shape ``(N, 3)`` with per-node 3D coordinates used to
            construct line segments.
        weights: 2D array of shape ``(N, N)`` with edge weights.
        top_frac_weights_displayed: Fraction in ``[0, 1]`` specifying how many
            of the heaviest edges to keep (approximately).
        global_weights_frac: If ``True``, the quantile threshold is computed
            from **all** upper-triangle weights. If ``False``, it is computed
            only from candidate edges (those with both endpoints in
            ``include``).
        global_opacity: If ``True``, opacity weights are derived from
            ``row_wise_norm(weights)``. If ``False``, they are derived only from
            the kept edges (others zeroed) before normalizing.
        global_color_saturation: If ``True``, color weights are derived from
            ``absolute_norm(weights)``. If ``False``, they are derived only from
            the kept edges (others zeroed) before normalizing.

    Returns:
        tuple:
            - ``line_segments`` (np.ndarray): Shape ``(E, 2, 3)`` where each row
              contains the ``[from_xyz, to_xyz]`` coordinates for one edge.
            - ``color_weights`` (np.ndarray): Shape ``(E,)`` scalar weights
              intended for colormap mapping (e.g., saturation).
            - ``opacity_weights`` (np.ndarray): Shape ``(E,)`` scalar weights
              intended for alpha/opacity.

    Raises:
        ValueError: If ``coords`` is not shape ``(N, 3)``.
    """
    _logger.debug(
        "build_line_segments: N=%s, include.len=%s, coords.shape=%s, weights.shape=%s, top_frac=%s, "
        "global_weights_frac=%s, global_opacity=%s, global_color_saturation=%s",
        N,
        None if include is None else np.asarray(include).size,
        getattr(coords, "shape", None),
        getattr(weights, "shape", None),
        top_frac_weights_displayed,
        global_weights_frac, global_opacity, global_color_saturation
    )

    # Candidate edges
    rows, cols = np.triu_indices(N, k=1)

    # Endpoint filter: keep edges whose BOTH endpoints are in 'include'
    inc_mask = np.zeros(N, dtype=bool)
    inc_idx = np.asarray(include, dtype=int)
    inc_mask[inc_idx] = True
    edge_mask = inc_mask[rows] & inc_mask[cols]
    rows, cols = rows[edge_mask], cols[edge_mask]

    if rows.size == 0:
        _logger.debug("build_line_segments: no candidate edges after endpoint filter; returning empties.")
        return (np.empty((0, 2, 3), dtype=float),
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=float))

    edge_weights = weights[rows, cols]

    # Threshold: global vs local (displayed-only) quantile
    if global_weights_frac:
        thresh = absolute_quantile(N, weights, top_frac_weights_displayed)
    else:
        thresh = float(np.quantile(edge_weights, 1.0 - top_frac_weights_displayed))
    kept = edge_weights >= thresh
    rows, cols = rows[kept], cols[kept]

    if rows.size == 0:
        _logger.debug("build_line_segments: no edges kept after threshold; returning empties.")
        return (np.empty((0, 2, 3), dtype=float),
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=float))

    # Build a matrix containing ONLY the kept edges (others zeroed)
    displayed_weights = np.zeros_like(weights)
    displayed_weights[rows, cols] = weights[rows, cols]
    displayed_weights[cols, rows] = weights[rows, cols]  # keep symmetry for row sums

    # Opacity weights: global vs displayed-only
    if global_opacity:
        opacity_weights = row_wise_norm(weights)[rows, cols]
    else:
        opacity_weights = row_wise_norm(displayed_weights)[rows, cols]

    # Color weights: global vs displayed-only (absolute normalization)
    if global_color_saturation:
        color_weights = absolute_norm(weights)[rows, cols]
    else:
        color_weights = absolute_norm(displayed_weights)[rows, cols]

    # Coordinates: EXPECT (N, 3)
    coords = np.asarray(coords)
    if coords.shape[0] != N:
        raise ValueError(
            f"`coords` must be shape (N, 3) with N={N}. "
            "If you spread only displayed nodes, create a copy of the full frame coords and "
            "overwrite those displayed rows before calling this function."
        )

    # Segments (E, 2, 3)
    line_segments = np.stack([coords[rows], coords[cols]], axis=1)

    _logger.debug("build_line_segments: segments.shape=%s, color_w.shape=%s, opacity_w.shape=%s, thresh=%.6g, kept=%d",
                  getattr(line_segments, "shape", None),
                  getattr(color_weights, "shape", None),
                  getattr(opacity_weights, "shape", None),
                  thresh, rows.size)

    return line_segments, color_weights, opacity_weights


__all__ = [
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
"SLATE",
"HEAT",
"COLD",
]

if __name__ == "__main__":
    pass
