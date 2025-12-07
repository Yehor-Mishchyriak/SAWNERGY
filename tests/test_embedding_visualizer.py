from __future__ import annotations

import numpy as np
import pytest

from sawnergy.embedding import visualizer as emb_vis_module

from .conftest import FRAME_COUNT, _DummyAxes, _DummyFigure, _DummyPlt, _DummyScatter, _DummyNormalize


def test_embedding_visualizer_dataflow(embeddings_archive_path, patched_embedding_visualizer):
    vis = emb_vis_module.Visualizer(embeddings_archive_path, show=False)
    # shapes loaded from archive
    assert vis.E.shape[0] == FRAME_COUNT
    assert vis.E.shape[1] > 0
    assert vis.E.shape[2] >= 2  # dimensionality >=2 in the fixture

    # build one frame; should update the dummy scatter without errors
    vis.build_frame(1, displayed_nodes="ALL", show_node_labels=False, show=False)
    # offsets3d set by build_frame
    x, y, z = vis._scatter._offsets3d
    assert len(x) == vis.N and len(y) == vis.N and len(z) == vis.N


def test_embedding_visualizer_labels_are_one_based(embeddings_archive_path, patched_embedding_visualizer):
    vis = emb_vis_module.Visualizer(embeddings_archive_path, show=False)

    captured: list[str] = []

    # capture labels through injected dummy text function
    def _capture_text(x, y, z, s, **_):
        captured.append(str(s))

    vis._ax.text = _capture_text  # type: ignore[attr-defined]

    picks = [1, 3]  # 1-based indices
    vis.build_frame(
        1,
        displayed_nodes=picks,
        show_node_labels=True,
        show=False,
    )

    assert captured and set(captured) == set(map(str, picks))


def test_embedding_visualizer_pca_3d_padding_when_dim2(embeddings_archive_path, patched_embedding_visualizer):
    """If embeddings are 2D, the Z coordinate must be zeros after PCA->3D projection."""
    vis = emb_vis_module.Visualizer(embeddings_archive_path, show=False)

    # Sanity: the synthetic embeddings in the fixture are D=2
    assert vis.E.shape[2] == 2

    vis.build_frame(1, displayed_nodes="ALL", show=False)
    _, _, z = vis._scatter._offsets3d
    z = np.asarray(z, dtype=float)
    assert z.shape == (vis.N,)
    assert np.allclose(z, 0.0)
