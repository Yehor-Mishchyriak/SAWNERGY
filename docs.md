# SAWNERGY Documentation

![LOGO](https://raw.githubusercontent.com/Yehor-Mishchyriak/SAWNERGY/main/assets/SAWNERGY_Logo_cropped.png)

A Python 3.11+ toolkit that converts molecular dynamics (MD) trajectories into residue interaction networks (RINs), samples random and self-avoiding walks (RW/SAW), trains skip-gram embeddings (PureML or PyTorch backends), and visualizes both networks and embeddings. All artifacts are Zarr v3 archives stored as compressed `.zip` files with rich metadata so every stage can be reproduced.
[![PyPI](https://img.shields.io/pypi/v/sawnergy)](https://pypi.org/project/sawnergy/) [![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/Yehor-Mishchyriak/SAWNERGY/blob/main/LICENSE) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) [![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/Yehor-Mishchyriak/SAWNERGY)

---

## Requirements
- Python >= 3.11.
- [`cpptraj`](https://ambermd.org/AmberTools.php) (AmberTools) discoverable on `PATH` **or** via `CPPTRAJ`, `AMBERHOME/bin`, or `CONDA_PREFIX/bin`. `RINBuilder` refuses to run without it.
- Default dependencies are installed via `pip install sawnergy`; PureML (`ym-pure-ml`) is bundled. PyTorch is optional but required for the `model_base="torch"` embedding backend (GPU if available).

## Installation
```bash
pip install sawnergy
# If you want the PyTorch backend, install torch separately:
# pip install torch
```

---

## Small visual example (constructed fully from trajectory and topology files)
![RIN](https://raw.githubusercontent.com/Yehor-Mishchyriak/SAWNERGY/main/assets/rin.png)
![Embedding](https://raw.githubusercontent.com/Yehor-Mishchyriak/SAWNERGY/main/assets/embedding.png)

## More visual examples:

### Animated Temporal Residue Interaction Network of Full Length p53 Protein
![RIN_animation](https://raw.githubusercontent.com/Yehor-Mishchyriak/SAWNERGY/main/assets/RIN_animation_compressed.gif)

### Residue Interaction Network of Full Length p53 Protein (on the right) and its Embedding (on the left)
![Embedding_vs_RIN](https://raw.githubusercontent.com/Yehor-Mishchyriak/SAWNERGY/main/assets/Embedding_vs_RIN_compressed.gif)

---

## End-to-End Quick Start (make sure cpptraj is discoverable)
```python
from pathlib import Path
import logging
import torch  # optional, only when using model_base="torch"

from sawnergy.logging_util import configure_logging
from sawnergy.rin import RINBuilder
from sawnergy.walks import Walker
from sawnergy.embedding import Embedder

configure_logging("./logs", file_level=logging.WARNING, console_level=logging.INFO, force=True)

# 1) Build a Residue Interaction Network archive
rin_path = Path("RIN_demo.zip")
rin_builder = RINBuilder()  # auto-locates cpptraj
rin_builder.build_rin(
    topology_file="system.prmtop",
    trajectory_file="trajectory.nc",
    molecule_of_interest=1,
    frame_range=(1, 100),          # inclusive, 1-based
    frame_batch_size=10,           # processed in 10-frame batches
    prune_low_energies_frac=0.9,   # per-row quantile pruning
    include_attractive=True,
    include_repulsive=False,
    output_path=rin_path,
)

# 2) Sample random / self-avoiding walks
walks_path = Path("WALKS_demo.zip")
with Walker(rin_path, seed=123) as walker:
    walker.sample_walks(
        walk_length=16,
        walks_per_node=100,
        include_attractive=True,
        include_repulsive=False,
        time_aware=False,
        output_path=walks_path,
        in_parallel=False,      # required kwarg; set True only under a main-guard
    )

# 3) Train per-frame embeddings
embedder = Embedder(walks_path, seed=999)
emb_path = embedder.embed_all(
    RIN_type="attr",              # "attr" or "repuls"
    using="merged",               # "RW" | "SAW" | "merged"
    num_epochs=20,
    negative_sampling=True,      # SG (False) or SGNS (True)
    num_negative_samples=10,
    window_size=5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_base="torch",
    kind="in",                    # stored embedding kind
    output_path="EMBEDDINGS_demo.zip",
)
print("Embeddings written to", emb_path)
```

```
MD Trajectory + Topology
          │
          ▼
      RINBuilder 
          │   →  RIN archive (.zip/.zarr) → Visualizer (display/animate RINs)
          ▼
        Walker
          │   →  Walks archive (RW/SAW per frame)
          ▼
       Embedder
          │   →  Embedding archive (frame × vocab × dim)
          ▼
     Downstream ML
```

Each stage consumes the archive produced by the previous one. Metadata embedded in the archives ensures frame order,
node indexing, and RNG seeds stay consistent across the toolchain.

---

## Archive Layouts (Zarr v3 in .zip)

All archives are Zarr v3 groups and can be opened directly with `sawnergy.sawnergy_util.ArrayStorage` (read via `mode="r"`; write/append via `mode="a"`/`"w"`). When compressed as `.zip`, they are **read-only**; create/append uses the `.zarr` directory form or a temporary store prior to compression.

| Archive | Core datasets (name → shape, dtype) | Key root attrs |
| --- | --- | --- |
| **RIN** | `ATTRACTIVE_transitions` `(T, N, N)` float32 (opt) • `REPULSIVE_transitions` `(T, N, N)` float32 (opt) • `ATTRACTIVE_energies` `(T, N, N)` float32 (opt; pre-normalized) • `REPULSIVE_energies` `(T, N, N)` float32 (opt) • `COM` `(T, N, 3)` float32 | `com_name="COM"` • `molecule_of_interest` • `frame_range` (tuple or None) • `frame_batch_size` • `prune_low_energies_frac` • `attractive_transitions_name` / `repulsive_transitions_name` (may be `None`) • `attractive_energies_name` / `repulsive_energies_name` (may be `None`) • `time_created` |
| **Walks** | `ATTRACTIVE_RWs` `(T, N·num_RWs, L+1)` uint16 (opt) • `REPULSIVE_RWs` `(T, N·num_RWs, L+1)` uint16 (opt) • `ATTRACTIVE_SAWs` `(T, N·num_SAWs, L+1)` uint16 (opt) • `REPULSIVE_SAWs` `(T, N·num_SAWs, L+1)` uint16 (opt) | `seed` • `num_workers` • `in_parallel` • `batch_size_nodes` • `num_RWs` • `num_SAWs` • `node_count` • `time_stamp_count` • `walk_length` • `walks_per_node` • dataset name attrs for each channel (may be `None`) • `walks_layout="time_leading_3d"` • `time_created` |
| **Embeddings** | `FRAME_EMBEDDINGS` `(T, N, D)` float32 | `frame_embeddings_name` • `time_stamp_count` • `node_count` • `embedding_dim` • `model_base` • `embedding_kind` (`"in"|"out"|"avg"`) • `objective` (`"sg"` or `"sgns"`) • `negative_sampling` • `num_epochs` • `batch_size` • `window_size` • `alpha` • `num_negative_samples` • `lr_step_per_batch` • `shuffle_data` • `device_hint` • `model_kwargs_repr` • `RIN_type` (`"attr"|"repuls"`) • `using` (`"RW"|"SAW"|"merged"`) • `source_WALKS_path` • walk metadata passthrough • `master_seed` • `per_frame_seeds` • `arrays_per_chunk` • `compression_level` • `created_at` |

`T` equals the number of frame **batches** produced by `RINBuilder` (i.e., `frame_range` swept in `frame_batch_size` steps). Walk node ids are **1-based** in storage; embedding training converts them to 0-based internally.

---

## Stage Reference

### RINBuilder (`sawnergy.rin.RINBuilder`)
- Purpose: run `cpptraj`, compute per-frame atomic EMAP/VMAP energies, project to residues, split into attractive/repulsive channels, prune, symmetrize, L1-normalize into transition matrices, and write a compressed archive.
- Construction: `RINBuilder(cpptraj_path=None)` auto-resolves `cpptraj` by checking an explicit path, `CPPTRAJ`, `PATH`, `AMBERHOME/bin`, then `CONDA_PREFIX/bin` (verification via `cpptraj -h`).
- `build_rin(...)` (returns output path as `str`):
  - Required: `topology_file`, `trajectory_file`, `molecule_of_interest` (e.g., `1` for `^1` mask in CPPTRAJ), `frame_range` (tuple or None).
  - Defaults: `frame_batch_size=-1` (all selected frames in one batch), `prune_low_energies_frac=0.85`, `keep_prenormalized_energies=True`, `include_attractive=True`, `include_repulsive=True`, `parallel_cpptraj=False`, `simul_cpptraj_instances=None` (→ `os.cpu_count()`), `num_matrices_in_compressed_blocks=10`, `compression_level=3`.
  - Processing per batch:
    1. `cpptraj pairwise` over the selected frame range → EMAP + VMAP → summed atomic interaction matrix (float32).
    2. Project to residues with the compositional mask `R = Pᵀ @ A @ P`.
    3. Split into channels (negative → attractive magnitude, positive → repulsive).
    4. Per-row quantile pruning at `q=prune_low_energies_frac` (applied independently to both channels).
    5. Zero diagonals, symmetrize `(M + Mᵀ)/2`, then row-wise L1 normalize (transition probabilities; breaks symmetry).
    6. Optionally persist pre-normalized energies; always persist normalized transitions for requested channels.
    7. Compute per-frame residue COMs for the batch and store their **batch mean** (so `T` equals the number of batches, not raw frames when `frame_batch_size>1`).
  - Output: compressed `.zip` (Zarr v3). Dataset names are recorded in attrs; absent channels are set to `None`. Types are float32 for matrices and COMs.
  - Notes: cpptraj tasks can be threaded (`parallel_cpptraj=True`); BLAS is kept single-threaded in that mode to avoid oversubscription.

### Walker (`sawnergy.walks.Walker`)
- Purpose: load **transition** matrices from a RIN archive into shared memory and sample RW/SAW paths (optionally time-aware).
- Construction: `Walker(RIN_path, seed=None)`; resolves transition dataset names from RIN attrs. Raises if neither channel exists or shapes are not `(T,N,N)` with matching `N`.
  - Shared memory: matrices live in `SharedNDArray`; call `close()` (or use `with Walker(...)`) to release and unlink segments.
  - RNG: a master seed is stored; child seeds per batch derive from `numpy.random.SeedSequence`.
- Key methods:
  - `_extract_prob_vector(node, time_stamp, interaction_type)` returns the transition row (float) for a node/time, renormalized after any masking.
  - `walk(start_node=None, start_time_stamp=None, length, interaction_type, self_avoid=False, time_aware=False, stickiness=None, on_no_options=None)` → `(length+1,)` array of **1-based** node ids. When `time_aware=True`, `stickiness` (probability of staying) and `on_no_options` (`"raise"` or `"loop"`) are mandatory; time steps are chosen by cosine similarity between transition matrices.
  - SAW dead-ends: if self-avoidance removes all probability mass for a step, the sampler logs a warning and falls back to an unconstrained RW move instead of raising.
  - `sample_walks(walk_length, walks_per_node, *, saw_frac=0.0, include_attractive=True, include_repulsive=False, time_aware=False, stickiness=None, on_no_options=None, output_path=None, in_parallel, max_parallel_workers=None, compression_level=3, num_walk_matrices_in_compressed_blocks=None)` → path to walks archive.
    - `in_parallel` is required (process-based via `ProcessPoolExecutor`); guard with `if __name__ == "__main__":` when `True`.
    - Per-node counts: `num_SAWs = round(walks_per_node * saw_frac)`, `num_RWs = walks_per_node - num_SAWs`.
    - Walk tensors are shaped `(T, total_walks, L+1)` with dtype `uint16`; time-aware walks live in the layer of their **start** time.
    - Metadata includes seeds, worker counts, and dataset names (`ATTRACTIVE_RWs`, `ATTRACTIVE_SAWs`, etc.; missing channels are stored as `None`).

### Embedder (`sawnergy.embedding.Embedder`)
- Purpose: turn walk corpora into skip-gram training pairs and fit embeddings per frame with either PureML (default) or PyTorch backends.
- Construction: `Embedder(WALKS_path, seed=None)`; loads available RW/SAW datasets and validates shapes against stored metadata. Walks are 1-based in storage and converted to 0-based internally.
- Single frame: `embed_frame(frame_id, RIN_type, using, num_epochs, *, negative_sampling=False, window_size=2, num_negative_samples=10, batch_size=1024, in_weights=None, out_weights=None, lr_step_per_batch=False, shuffle_data=True, dimensionality=128, alpha=0.75, device=None, model_base="pureml", model_kwargs=None, kind=("in",), _seed=None)` → list of `(embedding, kind)` tuples sorted as `avg`, `in`, `out`.
  - `RIN_type`: `"attr"` or `"repuls"`. `using`: `"RW"`, `"SAW"`, or `"merged"` (concatenates available RW and SAW for that channel).
  - Negative sampling toggles SGNS vs SG. Noise distributions are built **only** when `negative_sampling=True`.
  - Warm starts: `in_weights` shape `(V,D)` for all backends. `out_weights` shape `(V,D)` for SGNS; `(D,V)` for SG (the SG torch/PureML implementations transpose or expect that shape).
  - Raises if no pairs are produced or walks are missing.
- All frames: `embed_all(..., kind="in", output_path=None, num_matrices_in_compressed_blocks=20, compression_level=3)` → archive path.
  - Seeds: master seed stored; per-frame seeds derived deterministically and recorded in `per_frame_seeds`.
  - Warm start across frames: each frame initializes from the previous frame’s embeddings (`out` transposed for SG so shapes match).
  - Output dataset `FRAME_EMBEDDINGS` is float32 with shape `(T,N,D)`, where `T` is the walk archive’s `time_stamp_count`.

Backends:
- `model_base="pureml"` (default): uses `SGNS_PureML` or `SG_PureML` (no biases). Device hint is ignored (PureML is CPU-only).
- `model_base="torch"`: uses `SGNS_Torch` or `SG_Torch` (no biases). Defaults to CUDA if available unless `device` overrides it.

### Visualizers
- RIN Visualizer (`sawnergy.visual.Visualizer`):
  - Loads COM coordinates and optional attractive/repulsive energies from a RIN archive (dataset names resolved via attrs). If an energy channel is absent (`None`), that edge layer is skipped.
  - `build_frame(frame_id, displayed_nodes="ALL", displayed_pairwise_attraction_for_nodes="DISPLAYED_NODES", displayed_pairwise_repulsion_for_nodes="DISPLAYED_NODES", frac_node_interactions_displayed=0.01, global_interactions_frac=True, global_opacity=True, global_color_saturation=True, node_colors=None, title=None, padding=0.1, spread=1.0, show=False, *, show_node_labels=False, node_label_size=6, attractive_edge_cmap="autumn", repulsive_edge_cmap="winter")`.
    - Node selectors are 1-based; validation rejects non-integers before conversion.
    - Edge candidates are filtered to the specified nodes, then the heaviest fraction is drawn; color/opacity scaling can be global or restricted to kept edges.
  - `animate_trajectory(start=1, stop=None, step=1, interval_ms=50, loop=False, **build_kwargs)` reuses `build_frame`; enforces `show=False` during iteration and finally calls `show()` once for a single pass. Negative `step` plays backwards.
  - Backend handling: `ensure_backend(show)` picks a GUI backend; if `show=True` but no display is available, falls back to `Agg` and emits a warning.
- Embedding Visualizer (`sawnergy.embedding.Visualizer`):
  - Loads `FRAME_EMBEDDINGS` from an embeddings archive; optional `normalize_rows=True` L2-normalizes rows before PCA.
  - `build_frame(frame_id, *, node_colors="rainbow", displayed_nodes="ALL", show_node_labels=False, show=False)` projects the selected frame to 3D via SVD-based PCA (pads to 3 coordinates if `D<3`). Node selectors are 1-based and validated.
  - Shares color semantics with the RIN visualizer (`node_colors` can be a colormap string, per-node RGBA array shaped `(N,3|4)`, or group tuples).

### Utilities
- `ArrayStorage`: thin wrapper over Zarr v3 groups backed by a `.zarr` directory or read-only `.zip`. Handles per-block chunk metadata, JSON-safe attrs, block iteration, and compression via `compress(into=..., compression_level)` or context-managed `compress_and_cleanup(output_pth, compression_level)`.
  - `write` appends along axis 0; `read` and `block_iter` return NumPy arrays (copies). Default chunk length when unset is 10 with a warning.
  - Root attrs always include `array_chunk_size_in_block`, `array_shape_in_block`, and `array_dtype_in_block`.
- `logging_util.configure_logging(logs_dir, file_level=logging.WARNING, console_level=logging.WARNING, force=False)` installs a timed rotating file handler plus console handler. When `force=True`, existing root handlers are removed first.

---

## Practical Notes
- `Walker.sample_walks` uses process-based parallelism; wrap calls in `if __name__ == "__main__":` when `in_parallel=True`. Shared memory segments are unlinked only by the creating process; call `close()` (or use a context manager) to avoid leaks.
- Time-aware walks require `stickiness` in `[0,1]` and `on_no_options` set to `"raise"` or `"loop"`. Time transitions are chosen by cosine similarity between transition matrices, renormalized before sampling.
- Transition matrices coming from `RINBuilder` are row-normalized probabilities; `_step_node` renormalizes again after any avoidance masks to keep probabilities valid.
- Frame dimension `T` equals the number of **frame batches**, not necessarily the raw frame count if `frame_batch_size > 1`.
- Missing channels are represented by `None` in attrs; downstream stages skip them gracefully (e.g., repulsive walks/edges/embeddings are absent if never built).

---

## Minimal API Cheatsheet
- Build RIN: `RINBuilder().build_rin(topology_file, trajectory_file, molecule_of_interest, frame_range, frame_batch_size=-1, prune_low_energies_frac=0.85, keep_prenormalized_energies=True, include_attractive=True, include_repulsive=True, parallel_cpptraj=False, simul_cpptraj_instances=None, num_matrices_in_compressed_blocks=10, compression_level=3)`
- Walks: `Walker(rin_path, seed=None).sample_walks(walk_length, walks_per_node, saw_frac=0.0, include_attractive=True, include_repulsive=False, time_aware=False, stickiness=None, on_no_options=None, output_path=None, in_parallel=False, max_parallel_workers=None, compression_level=3, num_walk_matrices_in_compressed_blocks=None)`
- Embeddings: `Embedder(walks_path, seed=None).embed_all(RIN_type, using, num_epochs, negative_sampling=False, window_size=2, num_negative_samples=10, batch_size=1024, lr_step_per_batch=False, shuffle_data=True, dimensionality=128, alpha=0.75, device=None, model_base="pureml", model_kwargs=None, kind="in", output_path=None, num_matrices_in_compressed_blocks=20, compression_level=3)`
- Visualization: `sawnergy.visual.Visualizer(rin_path, show=False).build_frame(...)` or `.animate_trajectory(...)`; `sawnergy.embedding.Visualizer(emb_path, normalize_rows=False).build_frame(...)`.

All functions raise informative `ValueError`/`RuntimeError` when inputs are inconsistent (e.g., missing walks, out-of-range frame ids, invalid quantiles). Attributes recorded in each archive are intended to be sufficient to reproduce downstream stages without additional bookkeeping.
