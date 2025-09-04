# third-pary
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
# built-in
from typing import Iterable, Callable
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
HEAT = "gist_heat"
COLD = "winter"

# *----------------------------------------------------*
#                       FUNCTIONS
# *----------------------------------------------------*

# -=-=-=-=-=-=-=-=-=-=-=- #
#       CONVENIENCE     
# -=-=-=-=-=-=-=-=-=-=-=- #

def warm_start_matplotlib() -> None:
    """Prime font cache & 3D pipeline to avoid first-draw stalls."""
    try:
        from matplotlib import font_manager
        _ = font_manager.findSystemFonts()
        _ = font_manager.FontManager()
    except Exception:
        pass
    try:
        # tiny 3D figure + colormap + initial render
        f = plt.figure(figsize=(1, 1))
        ax = f.add_subplot(111, projection="3d")
        ax.plot([0, 1], [0, 1], [0, 1])
        f.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=ax, fraction=0.2, pad=0.04)
        f.canvas.draw_idle()
        plt.pause(0.01)
        plt.close(f)
    except Exception:
        pass

def map_groups_to_colors(N: int,
                        groups: dict[Iterable[int], str],
                        default_color: str,
                        one_based: bool = True):
    base = mcolors.to_rgba(default_color)
    colors = [base for _ in range(N)]
    for indices, hex_color in groups.items():
        col = mcolors.to_rgba(hex_color)
        for idx in indices:
            i = (idx - 1) if one_based else idx
            if not (0 <= i < N):
                raise IndexError(f"Index {idx} out of range for N={N}")
            colors[i] = col
    return colors

# -=-=-=-=-=-=-=-=-=-=-=- #
#    SCENE CONSTRUCTION     
# -=-=-=-=-=-=-=-=-=-=-=- #

def absolute_quantile(): # low energy cutoff
    pass

def row_wise_norm(): # opacity
    pass

def absolute_norm(): # color
    pass

def build_line_segments(
    N: int,
    include: np.ndarray, 
    coords: np.ndarray,
    weights: np.ndarray,
    thresh: float,
    colors: str,
    *,
    color_norm: Callable,
    opacity_norm: Callable):

    rows, cols = np.triu_indices(N, k=1) # skip the main diag

    rows = rows[np.isin(include)]
    cols = cols[np.isin(include)]

    edge_weights = weights[rows, cols] # 1D edge list
    kept = edge_weights > thresh
    rows, cols = rows[kept], cols[kept]

    # network data:
    edge_weights = edge_weights[kept]
    node_coords = coords[include, :]
    
    color_norm = Normalize(vmin=edge_weights.min(), vmax=edge_weights.max())
    opacity_norm = Normalize(vmin=edge_weights.min(), vmax=edge_weights.max())

    row_wise_opacity = 
    uniform_color_map = 

    segments = np.stack([node_coords[rows], node_coords[cols]], axis=1) # I'm a bit confused about masking here
    colors = ...
    opacities = ...

    return segments, colors, opacities
