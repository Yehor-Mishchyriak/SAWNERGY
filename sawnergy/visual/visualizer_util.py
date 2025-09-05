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

def warm_start_matplotlib() -> None:
    """Prime font cache & 3D pipeline to avoid first-draw stalls."""
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
                        groups: tuple[Iterable[int], str] | None,
                        default_color: str,
                        one_based: bool = True):
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
    _logger.debug("row_wise_norm: weights.shape=%s", getattr(weights, "shape", None))
    sums = np.sum(weights, axis=1, keepdims=True)
    out = weights / sums
    try:
        _logger.debug("row_wise_norm: row_sums[min=%.6g, max=%.6g]", float(sums.min()), float(sums.max()))
    except Exception:
        _logger.debug("row_wise_norm: row_sums stats unavailable.")
    return out

def absolute_norm(weights: np.ndarray) -> np.ndarray:
    _logger.debug("absolute_norm: weights.shape=%s", getattr(weights, "shape", None))
    total = np.sum(weights)
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
