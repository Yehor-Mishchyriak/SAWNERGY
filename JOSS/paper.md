---
title: "SAWNERGY: A Python framework for dynamic residue-interaction networks and walk-based embeddings from molecular dynamics simulations"
tags:
  - python
  - molecular-dynamics
  - bioinformatics
  - graph-embeddings
  - skip-gram
  - DeepWalk
  - zarr
authors:
  - name: Yehor Mishchyriak
    orcid: 0009-0001-8371-7159
    affiliation: "1, 2"
  - name: Kelly M. Thayer
    orcid: 0000-0001-7437-9517
    affiliation: "1, 2, 3, 4"
  - name: Sean Stetson
    orcid: 0009-0007-9759-5977
    affiliation: "1, 2, 3"
affiliations:
  - name: ThayerLab, Wesleyan University, Middletown, CT, United States
    index: 1
  - name: Department of Computer Science, Wesleyan University, Middletown, CT, United States
    index: 2
  - name: College of Integrative Sciences, Wesleyan University, Middletown, CT, United States
    index: 3
  - name: Department of Chemistry, Wesleyan University, Middletown, CT, United States
    index: 4
date: 5 December 2025
bibliography: paper.bib
---

# Summary
SAWNERGY is a Python toolkit that turns molecular dynamics (MD) trajectories into temporal residue interaction networks (TRINs), samples random walks, and trains DeepWalk-style skip-gram embeddings. It yields compact low-dimensional representations of residue interaction patterns, replacing bulky framewise adjacencies, and stores them as compressed Zarr archives with metadata for reproducibility and visualization.

# Statement of need
MD simulations yield high-dimensional pairwise data that are noisy and hard to compare over time; raw adjacencies scale as $O(N^2)$. Random-walk embeddings such as DeepWalk summarize multi-hop context in low-dimensional vectors and outperform linear projections like PCA on graph benchmarks [@perozzi2014deepwalk]. SAWNERGY packages this for biomolecular systems – RIN construction via `cpptraj`, walk sampling, and skip-gram training with PureML (NumPy) [@mishchyriak2025pureml] or PyTorch [@paszke2019pytorch] – so users get lightweight, reproducible tools for analysis and feature engineering without custom glue code.

# Software description

## RIN construction
Given a topology/trajectory pair, molecule ID, frame range, and batch size, SAWNERGY calls `cpptraj` [@roe2013cpptraj] to compute atomic electrostatic and van der Waals interaction matrices. Let $A \in \mathbb{R}^{n_a \times n_a}$ be the batch-averaged atomic matrix and $P \in \{0,1\}^{n_a \times n_r}$ map atoms to residues. The residue interaction matrix is
$$
R = P^{\top} A P \in \mathbb{R}^{n_r \times n_r}.
$$
This projection is needed because `cpptraj`’s `pairwise` driver reports atom–atom energies; residue-level interactions are not emitted directly.

SAWNERGY splits $R$ into attractive and repulsive channels:
$$
R^{-}_{ij} = \max(-R_{ij}, 0), \quad R^{+}_{ij} = \max(R_{ij}, 0),
$$
then prunes by per-row quantile, zeros self-interactions, symmetrizes $\tilde{R} = \tfrac{1}{2}(R + R^{\top})$, and row-normalizes to obtain a transition matrix $T$ with $\sum_j T_{ij} = 1$. Per-batch residue centers of mass are recorded. Outputs are chunked into Zarr v3 groups and can be compressed to read-only ZIP stores.

The pipeline emphasizes non-bonded electrostatics and van der Waals terms because they drive many inter-residue contacts (e.g., salt bridges, hydrogen bonds). They follow the Coulomb and Lennard-Jones forms,
$$
E_{\mathrm{elec}}(i,j) = \frac{q_i q_j}{4\pi\varepsilon_0 r_{ij}}, \qquad
E_{\mathrm{vdW}}(i,j) = 4\varepsilon_{ij}\left[\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right],
$$
and are reported by `cpptraj` at the atom-atom level. Aggregating weighted contacts over batches improves throughput at the cost of averaging within each batch, which can reduce temporal precision for fast dynamics.

## Walk sampling
Given a transition matrix $T$, a length-$L$ random walk has probability
$$
p(v_{0:L}) = \prod_{k=1}^{L} T_{v_{k-1}, v_k}.
$$
Self-avoiding walks enforce $v_k \notin \{v_0,\dots,v_{k-1}\}$ when possible. A time-aware variant augments the state with a time index $t$; with stickiness $s$, the walk remains at $t$ with probability $s$, otherwise samples a new $t'$ using cosine similarities between flattened transition matrices. Walks are stored as 3D arrays $(T, M, L{+}1)$ for each interaction channel.
Self-avoiding and time-aware walks are experimental; the standard DeepWalk-style setup relies on plain random walks. Users can mix in a fraction of self-avoiding walks, loosely analogous to node2vec’s $p,q$ biases [@grover2016node2vec]: plain walks revisit neighborhoods (BFS-like), while self-avoiding walks push outward (DFS-like).

## Embedding
SAWNERGY trains skip-gram (full softmax) or SGNS models over the walk corpora. For SGNS, given a positive pair $(u, v)$ and negatives $\mathcal{N}$, the loss minimizes
$$
\mathcal{L}_{\mathrm{SGNS}} = -\left[\log \sigma(\mathbf{u}^{\top}\mathbf{v}) + \sum_{n \in \mathcal{N}} \log \sigma(-\mathbf{u}^{\top}\mathbf{n})\right],
$$
learning embeddings $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$ that capture co-occurrence along walks (as in DeepWalk). For plain skip-gram with full softmax, the model minimizes
$$ 
\mathcal{L}_{\mathrm{SG}} = - \sum_{(u,v)} \log \frac{\exp(\mathbf{u}^{\top}\mathbf{v})}{\sum_{w \in V} \exp(\mathbf{u}^{\top}\mathbf{w})},
$$ 
training a single classifier over the vocabulary instead of using negative sampling. Both objectives yield compact vectors that encode interaction context more effectively than raw high-dimensional adjacency matrices. Per-frame embeddings are written as compressed blocks with metadata (vocabulary size, dimensions, seeds, training hyperparameters).
These embeddings capture interaction patterns between residues analogously to word embeddings capturing semantic proximity (e.g., “school” near “student”): residues that frequently co-occur along attractive/repulsive walks end up close in the learned space, reflecting structural context beyond direct pairwise links.

For cross-frame comparisons, SAWNERGY includes an orthogonal alignment helper (`align_frames`) that solves the Procrustes problem
$$
\min_{R \in O(d)} \| X R - Y \|_F \quad\Rightarrow\quad R = U V^{\top} \ \text{for}\ \mathrm{SVD}(X^{\top}Y) = U \Sigma V^{\top},
$$
with optional centering and reflection control, enabling post-hoc alignment of embeddings from different frames or runs.

## Implementation
- **RIN builder (`sawnergy.rin`)** orchestrates cpptraj calls, parses EMAP/VMAP blocks, projects to residues, applies pruning/symmetrization/normalization, and writes Zarr archives with metadata.
- **Walker (`sawnergy.walks`)** loads transition stacks into shared memory, samples random/self-avoiding/time-aware walks in parallel, frees the memory, and persists walk tensors.
- **Embedder (`sawnergy.embedding`)** loads walks, builds skip-gram/SGNS corpora, and trains with PureML (NumPy) [@mishchyriak2025pureml] or optional PyTorch backends [@paszke2019pytorch]. Frame alignment utilities are provided.
- **Visualizer (`sawnergy.visual`)** renders and animates temporal residue interaction networks (TRINs); per-frame embedding visualization is provided separately via `sawnergy.embedding.Visualizer`.
- **Storage** uses Zarr v3 with Blosc compression via the `ArrayStorage` helper; archives can be compressed to read-only ZIP for distribution.
- **Dependencies**: NumPy, Zarr, threadpoolctl, psutil, matplotlib, PureML (PyTorch optional) [@mishchyriak2025pureml; @paszke2019pytorch], and AmberTools `cpptraj` for RIN extraction.

# Quality control
The GitHub repository includes a `tests/` suite invoked via `pytest` covering storage helpers, walk sampling, embedding utilities, and math helpers. These tests run in continuous integration on each commit to the public repository and before PyPI releases to ensure reproducibility and stability. SAWNERGY is actively used within ThayerLab, including ongoing analyses of the 12 known p53 isoforms.

# Example usage
## 1. Build a RIN archive from an MD trajectory:
```python
from sawnergy.rin import RINBuilder
RINBuilder().build_rin(
    topology_file="topo.prmtop",
    trajectory_file="traj.nc",
    molecule_of_interest=1,
    frame_batch_size=10,
    prune_low_energies_frac=0.85,
    include_attractive=True,
    include_repulsive=False,
    num_matrices_in_compressed_blocks=10,
    compression_level=3,
    output_path="RIN.zip"
)
```
## 2. Sample random and self-avoiding walks:
```python
from sawnergy.walks import Walker
with Walker("RIN.zip") as w:
  w.sample_walks(
      walk_length=20,
      walks_per_node=100,
      saw_frac=0.25,
      include_attractive=True,
      include_repulsive=False,
      time_aware=False,
      in_parallel=False,
      output_path="WALKS.zip"
  )
```
## 3. Train embeddings (DeepWalk-style skip-gram/SGNS):
```python
from sawnergy.embedding import Embedder
emb = Embedder("WALKS.zip")
emb.embed_all(
    RIN_type="attr",
    using="merged",
    num_epochs=5,
    negative_sampling=True,
    window_size=5,
    num_negative_samples=10,
    dimensionality=128,
    model_base="pureml",
    shuffle_data=True,
    kind="in",
    output_path="EMBEDDINGS.zip"
)
```
## 4. Visualize per-frame embeddings:
```python
from sawnergy.embedding import Visualizer
viz = Visualizer("EMBEDDINGS.zip", normalize_rows=True)
viz.build_frame(15, show=True, show_node_labels=True)
```

## Recommendations

- RINs: keep `frame_batch_size` small for active systems; prune with care (`prune_low_energies_frac`) to avoid information loss.
- Walks: default to random walks; optionally mix a small fraction of self-avoiding walks for longer-range coverage.
- Embeddings: use PyTorch if GPU is available; otherwise PureML on CPU works well.

## Visual example:

### Residue Interaction Network of the Full Length p53 Tumor Suppressor Protein produced by SAWNERGY
![FL_p53_RIN](https://raw.githubusercontent.com/Yehor-Mishchyriak/SAWNERGY/main/assets/FL_p53_RIN.png)

### The above network embedded and visualized in 3D by SAWNERGY
![FL_p53_RIN](https://raw.githubusercontent.com/Yehor-Mishchyriak/SAWNERGY/main/assets/FL_p53_embedding.png)

## Potential applications
- Feature engineering for ML models informed by MD (e.g., embedding vectors as inputs for classification/regression tasks on stability, binding, or mutational effects).
- Conformational dynamics: clustering or dimensionality reduction on per-frame embeddings to identify states, transitions, and rare events.
- Comparative analysis: aligning embeddings across trajectories/conditions to quantify perturbations (mutations, ligands, pH/temperature changes).
- Visualization and interpretation: TRIN animations and embedding plots to communicate interaction changes over time.

# Availability
Source code: https://github.com/Yehor-Mishchyriak/SAWNERGY  
PyPI: https://pypi.org/project/sawnergy/  
Documentation: https://ymishchyriak.com/docs/SAWNERGY-DOCS  
License: Apache-2.0 (see `LICENSE`)

# Acknowledgements
The project was directed and supervised by Professor Kelly M. Thayer, MD simulations were produced by Sean Stetson, and the software was developed by Yehor Mishchyriak. We acknowledge the AmberTools/cpptraj community and the developers of PyTorch.

# References
