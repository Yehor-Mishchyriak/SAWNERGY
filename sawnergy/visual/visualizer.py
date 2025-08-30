from __future__ import annotations

# third-pary
import numpy as np
# built-in
from pathlib import Path
# local
from .. import sawnergy_util


class Visualizer:
    
    def __init__(self,
                RIN_path: str | Path,
                COM_dataset_name: str = "COM",
                attractive_interactions_dataset_name: str = "ATTRACTIVE",
                repulsive_interactions_dataset_name: str = "REPULSIVE") -> None:
        
        self.RIN_data = sawnergy_util.ArrayStorage(RIN_path, mode="r")
        self.COM_data_name: str = COM_dataset_name
        self.attrac_int_data_name: str = attractive_interactions_dataset_name
        self.repuls_int_data_name: str = repulsive_interactions_dataset_name

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
        frame_num: int,
        display_interactions: bool = True,
        quantile: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        frame_id = frame_num - 1 # 1-based
        
        pass

    # --------- PUBLIC ----------
    def visualize_frame(
        self,
        frame_num: int,
        display_interactions: bool = True, 
        quantile: float = 0.1,
        *,
        color_map: str = "nipy_spectral",
        figsize: tuple[int, int] = (10, 8),
        padding: float = 0.1,
        node_size: int = 120,
        edge_scale: float = 1.0,
        show_axes: bool = False,
    ) -> None:
        
        pass

    def visualize_trajectory():
        pass
