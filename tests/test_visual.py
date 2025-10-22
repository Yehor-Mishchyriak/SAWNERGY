from __future__ import annotations

import numpy as np

from sawnergy.visual import visualizer as visualizer_module

from .conftest import FRAME_COUNT, RESIDUE_COUNT


def test_visualizer_dataflow(rin_archive_path, patched_visualizer):
    vis = visualizer_module.Visualizer(rin_archive_path, show=False)
    assert vis.COM_coords.shape[0] == FRAME_COUNT
    assert vis.COM_coords.shape[1] == RESIDUE_COUNT
    assert vis.attr_energies is not None

    vis._update_scatter(vis.COM_coords[0])
    vis._update_attr_edges(np.zeros((0, 2, 3), dtype=float))
    vis._update_repuls_edges(np.zeros((0, 2, 3), dtype=float))
