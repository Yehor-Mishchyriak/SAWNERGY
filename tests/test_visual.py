from __future__ import annotations

import numpy as np
import pytest

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


def test_displayed_nodes_require_integers(rin_archive_path_both, patched_visualizer):
    vis = visualizer_module.Visualizer(rin_archive_path_both, show=False)
    with pytest.raises(TypeError):
        vis.build_frame(
            1,
            displayed_nodes=[1.5],
            displayed_pairwise_attraction_for_nodes=None,
            displayed_pairwise_repulsion_for_nodes=None,
            show=False,
        )


def test_real_visualizer_handles_empty_edge_collections(rin_archive_path_both, monkeypatch):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    add_collection3d = Axes3D.add_collection3d

    def autoscale_collection(self, collection, *args, **kwargs):
        # Reproduce CI's Matplotlib 3.11 empty-segment autoscaling failure on
        # older Matplotlib too, while retaining the real axes and artists.
        if isinstance(collection, Line3DCollection) and kwargs.get("autolim", True):
            np.concatenate(collection._segments3d)
        return add_collection3d(self, collection, *args, **kwargs)

    monkeypatch.setattr(Axes3D, "add_collection3d", autoscale_collection)
    monkeypatch.setattr(visualizer_module.Visualizer, "no_instances", False)
    before = set(plt.get_fignums())
    try:
        vis = visualizer_module.Visualizer(rin_archive_path_both, show=False)
        for artist in (vis._attr, vis._repuls):
            assert artist.axes is vis._ax and artist in vis._ax.collections
            assert len(artist._segments3d) == 0
        vis._fig.canvas.draw()
        vis.build_frame(1, frac_node_interactions_displayed=1.0, show=False)
        assert len(vis._attr._segments3d) > 0
        assert len(vis._repuls._segments3d) > 0
        vis._fig.canvas.draw()
        vis._update_attr_edges(np.empty((0, 2, 3)))
        vis._update_repuls_edges(np.empty((0, 2, 3)))
        vis._fig.canvas.draw()
    finally:
        for number in set(plt.get_fignums()) - before:
            plt.close(number)
