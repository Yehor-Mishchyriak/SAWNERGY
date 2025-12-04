---
title: "SAWNERGY: Residue interaction networks, random walks, and DeepWalk-style embeddings for molecular dynamics"
tags:
  - python
  - molecular-dynamics
  - bioinformatics
  - graph-embeddings
  - skip-gram
  - zarr
authors:
  - name: Yehor Mishchyriak
    orcid: 0009-0001-8371-7159
    affiliation: 1
  - name: Kelly M. Thayer
    orcid: 0000-0001-7437-9517
    affiliation: 1
  - name: Sean Stetson
    orcid: 0009-0007-9759-5977
    affiliation: 1
affiliations:
  - name: Thayer Lab
    index: 1
date: 2025-12-04
bibliography: paper.bib
---

# Summary
SAWNERGY is a Python toolkit that turns molecular dynamics (MD) trajectories into residue interaction networks (RINs), samples random and self-avoiding walks (including time-aware variants), and trains DeepWalk-style skip-gram embeddings over those walks. The workflow yields compact, nonlinearly learned low-dimensional representations of interaction patterns between amino acids, replacing the unwieldy high-dimensional adjacency tensors that arise directly from MD frames. Artifacts are persisted as compressed Zarr archives with metadata, enabling reproducibility and downstream visualization.

# Statement of need
MD simulations produce high-dimensional pairwise interaction data that are difficult to analyze or compare across time. Raw adjacency matrices scale as $O(N^2)$ with residue count and remain noisy across frames. Graph-based representations and random-walk embeddings such as DeepWalk provide a principled way to summarize interaction patterns: they preserve multi-hop context while yielding low-dimensional vectors suitable for clustering, alignment, and machine-learning tasks [@perozzi2014deepwalk]. SAWNERGY packages this pipeline end-to-end for biomolecular systems, including RIN construction from cpptraj outputs, walk sampling, and skip-gram/SGNS training with either PureML (NumPy) [@mishchyriak2025pureml] or PyTorch backends. The library targets computational biophysicists who need lightweight, reproducible tools to derive, store, and embed interaction graphs without building custom glue code.

# Software description

## RIN construction
Given a topology/trajectory pair, SAWNERGY calls `cpptraj` [@roe2013cpptraj] to compute per-frame electrostatic and van der Waals interaction matrices at the atomic level. Let $A \in \mathbb{R}^{n_a \times n_a}$ be the atomic interaction matrix for a frame, and $P \in \{0,1\}^{n_a \times n_r}$ map atoms to residues. The residue interaction matrix is
$$
R = P^{\top} A P \in \mathbb{R}^{n_r \times n_r}.
$$
SAWNERGY splits $R$ into attractive and repulsive channels:
$$
R^{-}_{ij} = \max(-R_{ij}, 0), \quad R^{+}_{ij} = \max(R_{ij}, 0),
$$
optionally prunes values below a per-row quantile, zeros self-interactions, symmetrizes $\tilde{R} = \tfrac{1}{2}(R + R^{\top})$, and row-normalizes to obtain a transition matrix $T$ satisfying $\sum_j T_{ij} = 1$. Per-batch centers of mass (COM) are also recorded. Outputs are chunked into Zarr v3 groups and can be compressed to read-only ZIP stores.

## Walk sampling
Given a transition matrix $T$, a length-$L$ random walk has probability
$$
p(v_{0:L}) = \prod_{k=1}^{L} T_{v_{k-1}, v_k}.
$$
Self-avoiding walks enforce $v_k \notin \{v_0,\dots,v_{k-1}\}$ when possible. A time-aware variant augments the state with a time index $t$; with stickiness $s$, the walk remains at $t$ with probability $s$, otherwise samples a new $t'$ using cosine similarities between transition slices. Walks are stored as 3D arrays $(T, M, L{+}1)$ for each interaction channel.

## Embedding
SAWNERGY trains skip-gram or SGNS models over the walk corpora. For a positive pair $(u, v)$ and negatives $\mathcal{N}$, the SGNS objective maximizes
$$
\log \sigma(\mathbf{u}^{\top}\mathbf{v}) + \sum_{n \in \mathcal{N}} \log \sigma(-\mathbf{u}^{\top}\mathbf{n}),
$$
learning embeddings $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$ that capture co-occurrence along walks (as in DeepWalk). This produces compact vectors that encode interaction context more effectively than raw high-dimensional adjacency matrices. Per-frame embeddings are written as compressed blocks with metadata (vocabulary size, dimensions, seeds, training hyperparameters).

## Implementation
- **RIN builder (`sawnergy.rin`)** orchestrates cpptraj calls, parses EMAP/VMAP blocks, projects to residues, applies pruning/symmetrization/normalization, and writes Zarr archives with metadata.
- **Walker (`sawnergy.walks`)** loads transition stacks into shared memory, samples random/self-avoiding/time-aware walks in parallel, and persists walk tensors.
- **Embedder (`sawnergy.embedding`)** loads walks, builds skip-gram/SGNS corpora, and trains with PureML (NumPy) [@mishchyriak2025pureml] or optional PyTorch backends. Frame alignment utilities are provided.
- **Visualizer (`sawnergy.visual`)** offers PCA-based 3D scatter plotting of per-frame embeddings.
- **Storage** uses Zarr v3 with Blosc compression; archives can be compressed to read-only ZIP for distribution.
- **Dependencies**: NumPy, Zarr, threadpoolctl, psutil, matplotlib, PureML (PyTorch optional) [@mishchyriak2025pureml], and AmberTools `cpptraj` for RIN extraction.

# Quality control
The repository includes a `tests/` suite invoked via `pytest` (see `CICD/test_runner.py`) covering storage helpers, walk sampling, embedding utilities, and math helpers. The CI script logs test output to `test_logs/`. Example scripts in `example_MD_for_quick_start/` and `example_analysis/` exercise the full pipeline on small datasets.

# Example usage
1. Build a RIN archive from an MD trajectory:
   ```bash
   python - <<'PY'
   from sawnergy.rin import RINBuilder
   RINBuilder().build_rin(
       topology_file="protein.prmtop",
       trajectory_file="traj.nc",
       molecule_of_interest=1,
       frame_batch_size=50,
       prune_low_energies_frac=0.85,
       include_attractive=True,
       include_repulsive=True,
       num_matrices_in_compressed_blocks=10,
       compression_level=3,
   )
   PY
   ```
2. Sample random and self-avoiding walks:
   ```bash
   python - <<'PY'
   from sawnergy.walks import Walker
   w = Walker("RIN_*.zip")
   w.sample_walks(
       walk_length=20,
       walks_per_node=8,
       saw_frac=0.5,
       include_attractive=True,
       include_repulsive=True,
       time_aware=False,
       in_parallel=False,
   )
   PY
   ```
3. Train embeddings (DeepWalk-style skip-gram/SGNS):
   ```bash
   python - <<'PY'
   from sawnergy.embedding import Embedder
   emb = Embedder("WALKS_*.zip")
   emb.embed_all(
       RIN_type="attr",
       using="merged",
       num_epochs=5,
       negative_sampling=True,
       window_size=2,
       num_negative_samples=5,
       dimensionality=128,
       model_base="pureml",
       kind="in",
   )
   PY
   ```
4. Visualize per-frame embeddings:
   ```bash
   python - <<'PY'
   from sawnergy.embedding import Visualizer
   vis = Visualizer("EMBEDDINGS_*.zip", show=False)
   vis.build_frame(frame_id=1, node_colors="rainbow", show_node_labels=False)
   vis.savefig("embedding_frame1.png")
   PY
   ```

# Acknowledgements
The project was directed and supervised by Kelly M. Thayer, MD simulations were produced by Sean Stetson, and the software was developed by Yehor Mishchyriak. We acknowledge the AmberTools/cpptraj community and the developers of PureML and PyTorch.

# References
