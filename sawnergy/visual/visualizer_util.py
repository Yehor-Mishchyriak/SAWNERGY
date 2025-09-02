# third-pary
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
# built-in
from typing import Iterable
import logging

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__) 

# *----------------------------------------------------*
#                       FUNCTIONS
# *----------------------------------------------------*

def create_weighted_edge_collection(coords: np.ndarray,
                        weights: np.ndarray,
                        thresh: float,
                        edge_scale: int,
                        colors: str) -> Line3DCollection:
    
    N = np.size(coords, axis=0)
    r, c = np.triu_indices(N, k=1) # skip the main diag
    kept = weights[r, c] > thresh
    r, c, edge_weights = r[kept], c[kept], weights[kept]
    segments = np.stack([coords[r], coords[c]], axis=1)
    norm = Normalize(vmin=edge_weights.min(), vmax=edge_weights.max())
    line_widths = edge_scale * norm(edge_weights)
    return Line3DCollection(segments, linewidths=line_widths, colors=colors)

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


if __name__ == "__main__":
    pass
