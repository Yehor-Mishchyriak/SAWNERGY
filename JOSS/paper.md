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
    affiliation: 1
  - name: Sean Stetson
    orcid: 0009-0007-9759-5977
    affiliation: "1, 2"
  - name: Kelly M. Thayer
    orcid: 0000-0001-7437-9517
    affiliation: "1, 2, 3"
affiliations:
  - name: Department of Computer Science, Wesleyan University, Middletown, CT, United States
    index: 1
  - name: College of Integrative Sciences, Wesleyan University, Middletown, CT, United States
    index: 2
  - name: Department of Chemistry, Wesleyan University, Middletown, CT, United States
    index: 3
date: 5 December 2025
bibliography: paper.bib
---

# Summary
SAWNERGY is a Python toolkit that turns molecular dynamics (MD) trajectories into temporal residue interaction networks (TRINs), samples random walks, and trains DeepWalk-style skip-gram embeddings. It yields compact low-dimensional representations of residue interaction patterns, replacing bulky framewise adjacencies, and stores them as compressed Zarr archives with metadata for reproducibility and visualization.

# Statement of need
MD simulations yield framewise pairwise interaction data that scales as $O(N^2)$ in the number of residues, and each residue’s interaction vector captures only its immediate neighborhood rather than the broader network context. The combination of high dimensionality and locality makes raw MD data cumbersome for analysis or machine learning. Yet long-range interaction patterns are essential for understanding allosteric effects caused by mutations or ligand binding, which is key in drug design. Hence, we need compact, context-rich representations to make MD-derived features usable.
A well-established solution is DeepWalk, a random-walk-based representation learning technique that summarizes multi-hop context in low-dimensional vectors and outperforms linear projections like PCA on graph benchmarks [@perozzi2014deepwalk].
To apply this to residue interaction networks, moving from raw weighted adjacencies to embeddings, one would need to glue together a large multi-stage workflow, which is error-prone, likely to be inefficient, and lack output reproducibility.
SAWNERGY adapts DeepWalk algorithm to weighted residue interaction graphs and packages the full pipeline from MD outputs to embeddings into a light Python framework. The framework is MD-format agnostic, involves parallel computation, post-execution clean-up, visualization and animation capabilities, data compression along with meta-data for reproducibilty, documentation, and tests.

# Interactions in question
SAWNERGY focuses on non-bonded interaction energies, namely electrostatic and van der Waals, computed from standard MD force fields [REF].
They follow the Coulomb and Lennard-Jones forms, respectively:
$$
E_{\mathrm{elec}}(i,j) = \frac{q_i q_j}{4\pi\varepsilon_0 r_{ij}}, \qquad
E_{\mathrm{vdW}}(i,j) = 4\varepsilon_{ij}\left[\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right].
$$
Using these equations `cpptraj` derives attractive and repulsive interaction energies between atoms in the system.

## Why electrostatic and van der Waals
Electrostatic and van der Waals interaction energies are the dominant non-bonded terms shaping residue–residue communication in folded proteins, and multiple studies from our group have shown that these quantities capture the functional reorganization of allosteric networks under mutation or ligand rescue. In p53 Y220C, electrostatic interaction networks differentiate native and mutant substates, reveal long-range communication pathways, and track shifts induced by allosteric effectors [@han2022insights; @han2024reconnaissance; @cowan2025network]. Energetic network comparisons also identify residues whose interaction patterns revert toward wild-type upon successful rescue, linking changes in local interaction energies to global structural response [@stetson2025restoration]. Across these studies, electrostatics and van der Waals contributions together provide a sensitive, low-level physical signal from which meaningful RINs can be constructed.
Additionally, these terms encode the energetic consequences of common inter-residue contacts, including salt bridges, hydrogen bonds, and packing interactions—since such contacts manifest as characteristic patterns in the underlying Coulomb and Lennard-Jones potentials.

# Pipeline description

## RIN construction
Given a topology/trajectory files and molecule ID in the system, SAWNERGY calls `cpptraj` [@roe2013cpptraj] to compute atomic interaction matrices, parsing EMAP/VMAP blocks, projecting to residues, pruning, symmetrizing, and normalizing to transitions, then writing Zarr archives with metadata.

Let $A \in \mathbb{R}^{n_a \times n_a}$ be the atomic matrix and $P \in \{0,1\}^{n_a \times n_r}$ map atoms to residues. The residue interaction matrix is
$$
R = P^{\top} A P \in \mathbb{R}^{n_r \times n_r}.
$$
This projection is needed because `cpptraj`’s `pairwise` driver reports atom–atom energies; residue-level interactions are not emitted directly.

SAWNERGY splits $R$ into attractive and repulsive interaction channels:
$$
R^{-}_{ij} = \max(-R_{ij}, 0), \quad R^{+}_{ij} = \max(R_{ij}, 0),
$$
then prunes by per-row quantile, zeros self-interactions, symmetrizes $\tilde{R} = \tfrac{1}{2}(R + R^{\top})$, and row-normalizes to obtain a transition matrix $T$ with $\sum_j T_{ij} = 1$.
Residue centers of mass are recorded for visualization.

Outputs are chunked into Zarr v3 groups and can be compressed to read-only ZIP stores.

For embeddings we recommend using the attractive channel, because stabilizing contacts (hydrogen bonds, salt bridges, hydrophobic packing) hold the fold together and define the meaningful co-occurrence structure along walks, whereas repulsive contributions are transient exclusions that add noise without encoding the binding network.

## Walk sampling
Given a transition matrix $T$ and length $L$, we treat residues as states in Markov process and draw walk sequences $v_{0:L}$ from $T$, starting at each residue in turn and recording the visited nodes. Transition stacks are loaded into shared memory so parallel workers sample without copies, and the sampler cleans up shared segments when done.

Self-avoiding walks enforce no node revisits.
SAWNERGY lets users mix in a fraction of SAWs (`saw_frac`) alongside plain random walks. This trade-off loosely mirrors node2vec’s $p,q$ biases [@grover2016node2vec]: plain walks revisit neighborhoods (BFS-like), while higher `saw_frac` encourages exploration of more distant regions of the graph like DFS.

## Embedding
SAWNERGY trains skip-gram (full softmax) or SGNS models over the sequences of random walk visits to predict pairs of co-occuring residues. For SGNS, given a true pair $(u, v)$ from a walk sample and set of random pairs $\mathcal{N}$ sampled from a distribution proportional to frequency counts across all the walks, we use gradient descent to minimize the following loss
$$
\mathcal{L}_{\mathrm{SGNS}} = -\left[\log \sigma(\mathbf{u}^{\top}\mathbf{v}) + \sum_{n \in \mathcal{N}} \log \sigma(-\mathbf{u}^{\top}\mathbf{n})\right],
$$
learning embeddings $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$. For plain skip-gram with full softmax, the model minimizes
$$ 
\mathcal{L}_{\mathrm{SG}} = - \sum_{(u,v)} \log \frac{\exp(\mathbf{u}^{\top}\mathbf{v})}{\sum_{w \in V} \exp(\mathbf{u}^{\top}\mathbf{w})},
$$ 
training a single classifier over the vocabulary instead of using negative sampling. Both objectives yield compact vectors that encode interaction context.

For cross-frame comparisons, SAWNERGY includes an orthogonal alignment helper (`align_frames`) that solves the Procrustes problem
$$
\min_{R \in O(d)} \| X R - Y \|_F \quad\Rightarrow\quad R = U V^{\top} \ \text{for}\ \mathrm{SVD}(X^{\top}Y) = U \Sigma V^{\top},
$$
with optional centering and reflection control, enabling post-hoc alignment of embeddings from different frames or runs.

Training backends include PureML (NumPy) [@mishchyriak2025pureml] and optional PyTorch [@paszke2019pytorch]. Per-frame embeddings are stored in the same compressed Zarr format with metadata; RINs and embeddings can be visualized via `sawnergy.visual.Visualizer` and `sawnergy.embedding.Visualizer`.

*Note: these steps are performed for every frame or batch of frames, with in-batch interaction averaging during TRIN construction, specified via `frame_batch_size`.*

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

## Visual example produced by SAWNERGY

![RIN of p53 Tumor Suppressor Protein](https://raw.githubusercontent.com/Yehor-Mishchyriak/SAWNERGY/main/assets/FL_p53_RIN.png)

![Embedding of p53 Tumor Suppressor Protein Projected onto 3 leading PCs](https://raw.githubusercontent.com/Yehor-Mishchyriak/SAWNERGY/main/assets/FL_p53_embedding.png)

*Note the visual resemblance of the two images despite the first one being center of mass coordinates of amino acid residues and the second one having been derived purely from random walks over the energetic network*

## Potential applications (TO BE EDITED OR MOVED)
- Feature engineering for ML models informed by MD (e.g., embedding vectors as inputs for classification/regression tasks on stability, binding, or mutational effects).
- Conformational dynamics: clustering or dimensionality reduction on per-frame embeddings to identify states, transitions, and rare events.
- Comparative analysis: aligning embeddings across trajectories/conditions to quantify perturbations (mutations, ligands, pH/temperature changes).
- Visualization and interpretation: TRIN animations and embedding plots to communicate interaction changes over time.

# Availability
Source code: https://github.com/Yehor-Mishchyriak/SAWNERGY  
PyPI: https://pypi.org/project/sawnergy/  
Documentation: https://ymishchyriak.com/docs/SAWNERGY-DOCS  
License: Apache-2.0 (see `LICENSE`)

# Acknowledgements (TO BE EDITED)
Supported by NSF CNS-0619508 and CNS-095985 (high-performance computing facilities at Wesleyan), NIH R15 GM128102-02, and NSF CHE-2320718. We thank Henk Meij for technical assistance with the HPC cluster. We also acknowledge the AmberTools/cpptraj and PyTorch communities.

# References
