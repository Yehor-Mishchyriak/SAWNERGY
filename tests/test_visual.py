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

def test_labels_are_one_based(rin_archive_path_both, patched_visualizer):
    vis = visualizer_module.Visualizer(rin_archive_path_both, show=False)
    captured = []
    # Inject a 'text' method to capture labels
    def _capture_text(x, y, z, s, **_):
        captured.append(str(s))
    vis._ax.text = _capture_text  # type: ignore[attr-defined]

    # Request labels for specific 1-based nodes
    picks = [1, 3]
    vis.build_frame(
        1,
        displayed_nodes=picks,
        displayed_pairwise_attraction_for_nodes="DISPLAYED_NODES",
        displayed_pairwise_repulsion_for_nodes="DISPLAYED_NODES",
        show_node_labels=True,
        node_label_size=5,
        show=False,
    )
    assert captured and set(captured) == set(map(str, picks))

def test_repulsive_edges_available(rin_archive_path_both, patched_visualizer):
    vis = visualizer_module.Visualizer(rin_archive_path_both, show=False)
    assert vis.repuls_energies is not None
    # Ensure update path does not crash and records segments
    vis._update_repuls_edges(np.zeros((0, 2, 3)), colors=np.zeros((0, 4)), opacity=None)

