# SAWNERGY

[![PyPI](https://img.shields.io/pypi/v/sawnergy)](https://pypi.org/project/sawnergy/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/Yehor-Mishchyriak/SAWNERGY/blob/main/LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
[![Docs](https://img.shields.io/badge/docs-available-blue)](https://ymishchyriak.com/docs/SAWNERGY-DOCS)

![LOGO](https://raw.githubusercontent.com/Yehor-Mishchyriak/SAWNERGY/main/assets/SAWNERGY_Logo_cropped.png)

A toolkit for transforming molecular dynamics (MD) trajectories into rich graph representations, sampling
random and self-avoiding walks, learning node embeddings, and visualizing residue interaction networks (RINs). SAWNERGY
keeps the full workflow — from `cpptraj` output to skip-gram embeddings (DeepWalk approach) — inside Python, backed by efficient Zarr-based archives and optional GPU acceleration.

---

## Installation

   ```bash
   pip install sawnergy
   ```

> **Optional:** For GPU training, install PyTorch separately (e.g., `pip install torch`).
> **Note:** RIN building requires `cpptraj` (AmberTools). Ensure it is discoverable via `$PATH` or the `CPPTRAJ`
> environment variable. Probably the easiest solution: install AmberTools via Conda, activate the environment, and SAWNERGY will find the cpptraj executable on its own, so just run your code and don't worry about it.

---

# UPDATES:

## v1.1.5 — What’s new:
- **Embedding visualizer color API.** `sawnergy.embedding.Visualizer` now accepts the same group/color tuples as the RIN visualizer (e.g., `(indices, sawnergy.visual.BLUE)`), so embedding plots and RIN plots share a unified coloring interface.

## v1.1.4 — What’s new:
- **SAW dead-end guard.** When self-avoidance zeroes out a transition row, the walker now logs a warning and takes an unconstrained RW step instead of raising, so sampling runs finish even on disconnected nodes.

## v1.1.3 — What’s new:
- Added more visual examples into the README
- Dedicated docs site is live: https://ymishchyriak.com/docs/SAWNERGY-DOCS (mirrors this repo and stays current).

## v1.1.2 — What’s new:
- **Safer warm starts across backends.**
  - Torch: Skip-Gram (full softmax) now transposes provided out warm-starts before copying, matching Linear’s `(out_features, in_features)` layout.
  - PureML: Both SGNS and SG defensively copy/validate warm-start arrays (correct shapes, immutable after construction); SG also transposes `(D, V)` out weights to `(V, D)` for embedding access.

## v1.1.1 — What’s new:
- **Walker parallelism is configurable.**
  - `Walker.sample_walks(..., in_parallel=True)` now accepts `max_parallel_workers` so you can lower the worker count below `os.cpu_count()` when sharing machines or reserving cores for other workloads.

## v1.1.0 — What’s new:
- **Logging helper respects `force` resets.**
  - `configure_logging()` now documents the correct defaults, and an optional `force=True` clears existing handlers before installing fresh ones — useful for scripts/notebooks that reconfigure logging multiple times.
- **`ArrayStorage` is easier to introspect.**
  - Added a readable `__repr__` plus `list_blocks()` so you can quickly inspect the stored datasets when debugging archives or working interactively.
- **Visualizer selectors are safer and lighter.**
  - `displayed_nodes` (and related selectors) now reject non-integer inputs before converting to 0-based indices, and edge coordinate buffers are only materialized when an edge layer is requested, reducing unnecessary copies when plotting nodes only.
- **Walker sampling is more robust.**
  - Transition rows are renormalized before RNG sampling (even without avoidance sets), and walk paths are accumulated in preallocated arrays, keeping long walks numerically stable and memory efficient.
- **Training prep and tooling tweaks.**
  - Skip-gram runs skip building noise distributions entirely (SGNS still gets them), cutting redundant `np.bincount`/normalization work, and `locate_cpptraj()` now de-duplicates candidate paths before probing to avoid repeated `cpptraj -h` calls.

## v1.0.9 — What’s new:
- **`SGNS_Torch` is no longer deprecated.**
  - The root cause was weight initialization; it’s fixed.
- **`SG_Torch` and `SG_PureML` no longer use biases.**
  - Affine/Linear layers no longer translate embeddings away from the origin.
- **Warm starts for frame embeddings.**
  - Each frame initializes from the preceding frame’s representation. This speeds convergence and keeps the basis approximately consistent.
- **Alignment function for comparing embeddings from different latent spaces.**
  - Based on the Orthogonal Procrustes solution: finds the best-fit orthogonal map between two embedding sets. Orthogonality preserves angles and relative distances, enabling direct comparison across bases.

## v1.0.8 — What’s new:
- **Temporary deprecation of `SGNS_Torch`**
  - `sawnergy.embedding.SGNS_Torch` currently produces noisy embeddings in practice. The issue likely stems from **weight initialization**, although the root cause has not yet been conclusively determined.
  - **Action:** The class and its `__init__` docstring now carry a deprecation notice. Constructing the class emits a **`DeprecationWarning`** and logs a **warning**.
  - **Use instead:** Prefer **`SG_Torch`** (plain Skip-Gram with full softmax) or the PureML backends **`SGNS_PureML`** / **`SG_PureML`**.
  - **Compatibility:** No breaking API changes; imports remain stable. PureML backends are unaffected.
- **Embedding visualizer update**
  - Now you can L2 normalize your embeddings before display.
- **Small improvements in the embedding module**
  - Improved API with a lot of good defaults in place to ease usage out of the box.
  - Small internal model tweaks.

## v1.0.7 — What’s new:
- **Added plain Skip-Gram model**
  - Now, the user can choose if they want to apply the negative sampling technique (two binary classifiers) or train a single classifier over the vocabulary (full softmax). For more detail, see: [deepwalk](https://arxiv.org/pdf/1403.6652), [word2vec](https://arxiv.org/pdf/1301.3781), and [negative_sampling](https://arxiv.org/pdf/1402.3722).
- **Set a harsher default for low interaction energies pruning during RIN construction**
  - Now we zero out 85% of the lowest interaction energies as opposed to the past 30% default, leading to more meaningful embeddings.
- **BUG FIX: Visualizer**
  - Previously, the visualizer would silently draw edges of 0 magnitude, meaning they were actually being drawn but were invisible due to full transparency and 0 width. As a result, the displayed image/animation would be very laggy. Now, this was fixed, and given the higher pruning default, the displayed interaction networks are clean and smooth under rotations, dragging, etc.
- **New Embedding Visualizer (3D)**
  - New lightweight viewer for per-frame embeddings that projects embeddings with PCA to a **3D** scatter. Supports the same node coloring semantics, optional node labels, and the same antialiasing/depthshade controls. Works in headless setups using the same backend guard and uses a blocking `show=True` for scripts.

---

## Why SAWNERGY?

- **Bridge simulations and graph ML**: Convert raw MD trajectories into residue interaction networks ready for graph
  algorithms and downstream machine learning tasks.
- **Deterministic, shareable artifacts**: Every stage produces compressed Zarr archives that contain both data and metadata so runs can be reproduced, shared, or inspected later.
- **High-performance data handling**: Heavy arrays live in shared memory during walk sampling to allow parallel processing without serialization overhead; archives are written in chunked, compressed form for fast read/write.
- **Flexible objectives & backends**: Train Skip-Gram with **negative sampling** (`objective="sgns"`) or **plain Skip-Gram** (`objective="sg"`), using either **PureML** (default) or **PyTorch**.
- **Visualization out of the box**: Plot and animate residue networks without leaving Python, using the data produced by RINBuilder.

---

## Pipeline at a Glance

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

## Small visual example (constructed fully from trajectory and topology files)
![RIN](https://raw.githubusercontent.com/Yehor-Mishchyriak/SAWNERGY/main/assets/rin.png)
![Embedding](https://raw.githubusercontent.com/Yehor-Mishchyriak/SAWNERGY/main/assets/embedding.png)

## Quick-start MD example

A minimal dataset is included in `example_MD_for_quick_start/` on GitHub to let you run the full SAWNERGY pipeline immediately:

- `p53_DBD.prmtop` (topology), `p53_DBD.pdb` (reference), `p53_DBD.nc` (trajectory)
- 1 µs production trajectory of the p53 DNA-binding domain, 1000 snapshots saved every 1 ns
- Credits: MD simulation produced by Sean Stetson (ORCID: 0009-0007-9759-5977)
- Intended use: quick-start tutorial for building RINs, sampling walks, and training embeddings without setting up your own MD run

See `example_MD_for_quick_start/brief_description.md`.

## More visual examples:

### Animated Temporal Residue Interaction Network of Full Length p53 Protein
![RIN_animation](https://raw.githubusercontent.com/Yehor-Mishchyriak/SAWNERGY/main/assets/RIN_animation_compressed.gif)

### Residue Interaction Network of Full Length p53 Protein (on the right) and its Embedding (on the left)
![Embedding_vs_RIN](https://raw.githubusercontent.com/Yehor-Mishchyriak/SAWNERGY/main/assets/Embedding_vs_RIN_compressed.gif)

---

## Core Components

### `sawnergy.rin.RINBuilder`

* Wraps the AmberTools `cpptraj` executable to:
  - compute per-frame electrostatic (EMAP) and van der Waals (VMAP) energy matrices at the atomic level,
  - project atom–atom interactions to residue–residue interactions using compositional masks,
  - prune, symmetrize, remove self-interactions, and L1-normalize the matrices,
  - compute per-residue centers of mass (COM) over the same frames.
* Outputs a compressed Zarr archive with transition matrices, optional pre-normalized energies, COM snapshots, and rich
  metadata (frame range, pruning quantile, molecule ID, etc.).
* Supports parallel `cpptraj` execution, batch processing, and keeps temporary stores tidy via
  `ArrayStorage.compress_and_cleanup`.

### `sawnergy.visual.Visualizer`

* Opens RIN archives, resolves dataset names from attributes, and renders nodes plus attractive/repulsive edge bundles
  in 3D using Matplotlib.
* Allows both static frame visualization and trajectory animation.
* Handles backend selection (`Agg` fallback in headless environments) and offers convenient color palettes via
  `visualizer_util`.

### `sawnergy.walks.Walker`

* Attaches to the RIN archive and loads attractive/repulsive transition matrices into shared memory using
  `walker_util.SharedNDArray` so multiple processes can sample without copying.
* Samples random walks (RW) and self-avoiding walks (SAW), optionally time-aware, that is, walks move through transition matrices with transition probabilities proportional to cosine similarity between the current and next frame. Randomness is controlled by the seed passed to the class constructor.
* Persists walks as `(time, walk_id, length+1)` tensors (1-based node indices) alongside metadata such as
  `walk_length`, `walks_per_node`, and RNG scheme.

### `sawnergy.embedding.Embedder`

* Consumes walk archives, generates skip-gram pairs, and normalizes them to 0-based indices.
* Selects skip-gram (SG / SGNS) backends dynamically via `model_base="pureml"|"torch"` with per-backend overrides supplied through `model_kwargs`.
* Handles deterministic per-frame seeding and returns the requested embedding `kind` (`"in"`, `"out"`, or `"avg"`) from `embed_frame` and `embed_all`.
* Persists per-frame matrices with rich provenance (walk metadata, objective, hyperparameters, RNG seeds) when `embed_all` targets an output archive.

### Supporting Utilities

* `sawnergy.sawnergy_util`
  - `ArrayStorage`: thin wrapper over Zarr v3 with helpers for chunk management, attribute coercion to JSON, and transparent compression to `.zip` archives.
  - Parallel helpers (`elementwise_processor`, `compose_steps`, etc.), temporary file management, logging, and runtime
    inspection utilities.
* `sawnergy.logging_util.configure_logging`: configure rotating file/console logging consistently across scripts.

---

## Archive Layouts

| Archive | Key datasets (name → shape, dtype) | Important attributes (root `attrs`) |
|---|---|---|
| **RIN** | `ATTRACTIVE_transitions` → **(T, N, N)**, float32  •  `REPULSIVE_transitions` → **(T, N, N)**, float32 (optional)  •  `ATTRACTIVE_energies` → **(T, N, N)**, float32 (optional)  •  `REPULSIVE_energies` → **(T, N, N)**, float32 (optional)  •  `COM` → **(T, N, 3)**, float32 | `time_created` (ISO) • `com_name` = `"COM"` • `molecule_of_interest` (int) • `frame_range` = `(start, end)` inclusive • `frame_batch_size` (int) • `prune_low_energies_frac` (float in [0,1]) • `attractive_transitions_name` / `repulsive_transitions_name` (dataset names or `None`) • `attractive_energies_name` / `repulsive_energies_name` (dataset names or `None`) |
| **Walks** | `ATTRACTIVE_RWs` → **(T, N·num_RWs, L+1)**, int32 (optional)  •  `REPULSIVE_RWs` → **(T, N·num_RWs, L+1)**, int32 (optional)  •  `ATTRACTIVE_SAWs` → **(T, N·num_SAWs, L+1)**, int32 (optional)  •  `REPULSIVE_SAWs` → **(T, N·num_SAWs, L+1)**, int32 (optional)  <br/>_Note:_ node IDs are **1-based**.| `time_created` (ISO) • `seed` (int) • `rng_scheme` = `"SeedSequence.spawn_per_batch_v1"` • `num_workers` (int) • `in_parallel` (bool) • `batch_size_nodes` (int) • `num_RWs` / `num_SAWs` (ints) • `node_count` (N) • `time_stamp_count` (T) • `walk_length` (L) • `walks_per_node` (int) • `attractive_RWs_name` / `repulsive_RWs_name` / `attractive_SAWs_name` / `repulsive_SAWs_name` (dataset names or `None`) • `walks_layout` = `"time_leading_3d"` |
| **Embeddings** | `FRAME_EMBEDDINGS` → **(T, N, D)**, float32 | `created_at` (ISO) • `frame_embeddings_name` = `"FRAME_EMBEDDINGS"` • `time_stamp_count` = T • `node_count` = N • `embedding_dim` = D • `model_base` = `"torch"` or `"pureml"` • `embedding_kind` = `"in"|"out"|"avg"` • `objective` = `"sgns"` or `"sg"` • `negative_sampling` (bool) • `num_negative_samples` (int) • `num_epochs` (int) • `batch_size` (int) • `window_size` (int) • `alpha` (float) • `lr_step_per_batch` (bool) • `shuffle_data` (bool) • `device_hint` (str) • `model_kwargs_repr` (repr string) • `RIN_type` = `"attr"` or `"repuls"` • `using` = `"RW"|"SAW"|"merged"` • `source_WALKS_path` (str) • `walk_length` (int) • `num_RWs` / `num_SAWs` (ints) • `attractive_*_name` / `repulsive_*_name` (dataset names or `None`) • `master_seed` (int) • `per_frame_seeds` (list[int]) • `arrays_per_chunk` (int) • `compression_level` (int) |

**Notes**

- In **RIN**, `T` equals the number of frame **batches** written (i.e., `frame_range` swept in steps of `frame_batch_size`). `ATTRACTIVE/REPULSIVE_energies` are **pre-normalized** absolute energies (written only when `keep_prenormalized_energies=True`), whereas `ATTRACTIVE/REPULSIVE_transitions` are the **row-wise L1-normalized** versions used for sampling.
- All archives are Zarr v3 groups. ArrayStorage also maintains per-block metadata in root attrs: `array_chunk_size_in_block`, `array_shape_in_block`, and `array_dtype_in_block` (dicts keyed by dataset name). You’ll see these in every archive.
- In **Embeddings**, `alpha` and `num_negative_samples` apply to **SGNS** only and are ignored for `objective="sg"`.

---

## Quick Start

```python
from pathlib import Path
from sawnergy.logging_util import configure_logging
from sawnergy.rin import RINBuilder
from sawnergy.walks import Walker
from sawnergy.embedding import Embedder

import logging
configure_logging("./logs", file_level=logging.WARNING, console_level=logging.INFO)

# 1. Build a Residue Interaction Network archive
rin_path = Path("./RIN_demo.zip")
rin_builder = RINBuilder()
rin_builder.build_rin(
    topology_file="system.prmtop",
    trajectory_file="trajectory.nc",
    molecule_of_interest=1,
    frame_range=(1, 100),
    frame_batch_size=10,
    prune_low_energies_frac=0.85,
    output_path=rin_path,
    include_attractive=True,
    include_repulsive=False
)

# 2. Sample walks from the RIN
walker = Walker(rin_path, seed=123)
walks_path = Path("./WALKS_demo.zip")
walker.sample_walks(
    walk_length=16,
    walks_per_node=100,
    saw_frac=0.25,
    include_attractive=True,
    include_repulsive=False,
    time_aware=False,
    output_path=walks_path,
    in_parallel=False
)
walker.close()

# 3. Train embeddings per frame (PyTorch backend)
import torch

embedder = Embedder(walks_path, seed=999)
embeddings_path = embedder.embed_all(
    RIN_type="attr",
    using="merged",
    num_epochs=10,
    negative_sampling=False,
    window_size=4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_base="torch",
    output_path="./EMBEDDINGS_demo.zip"
)
print("Embeddings written to", embeddings_path)
```

> For the PureML backend, set `model_base="pureml"` and pass the optimizer / scheduler classes inside `model_kwargs`.

---

## Visualization

```python
from sawnergy.visual import Visualizer

v = Visualizer("./RIN_demo.zip")
v.build_frame(1,
    node_colors="rainbow",
    displayed_nodes="ALL",
    displayed_pairwise_attraction_for_nodes="DISPLAYED_NODES",
    displayed_pairwise_repulsion_for_nodes="DISPLAYED_NODES",
    show_node_labels=True,
    show=True
)
```

`Visualizer` lazily loads datasets and works even in headless environments (falls back to the `Agg` backend).

```python
from sawnergy.embedding import Visualizer

viz = Visualizer("./EMBEDDINGS_demo.zip", normalize_rows=True)
viz.build_frame(1, show=True)
```

---

## Advanced Notes

- **Time-aware walks**: Set `time_aware=True`, provide `stickiness` and `on_no_options` when calling `Walker.sample_walks`.
- **Shared memory lifecycle**: Call `Walker.close()` (or use a context manager) to release shared-memory segments.
- **PureML vs PyTorch**: Select the backend at call time with `model_base="pureml"|"torch"` (defaults to `"pureml"`) and pass optimizer / scheduler overrides through `model_kwargs`.
- **ArrayStorage utilities**: Use `ArrayStorage` directly to peek into archives, append arrays, or manage metadata.

---

## Project Structure

```
├── sawnergy/
│   ├── rin/           # RINBuilder and cpptraj integration helpers
│   ├── walks/         # Walker class and shared-memory utilities
│   ├── embedding/     # Embedder + SG/SGNS backends (PureML / PyTorch)
│   ├── visual/        # Visualizer and palette utilities
│   │
│   ├── logging_util.py
│   └── sawnergy_util.py
│
└── README.md
```

---

## Contributing

Issues, enhancement suggestions, and discussions are always welcome!
Also, please tell your friends about the project!

A quick note:
Currently, the repository is view-only and updated only through a CI/CD pipeline connected to a private development repository.
Unfortunately, this means that if you submit a pull request and it gets merged, you won’t receive contributor credit on GitHub — which I know isn’t ideal.

That said (!), if you contribute via a PR at this stage, you’ll be permanently credited in both CREDITS.md and README.md.
I promise that as the project grows and I start relying more on community contributions, I’ll fix this by setting up a proper CI/CD workflow via GitHub Actions,
so everyone gets visible and fair credit for their work.

Thank you, and apologies for the inconvenience!

## Acknowledgments

SAWNERGY builds on the AmberTools `cpptraj` ecosystem, NumPy, Matplotlib, Zarr, and PyTorch (for GPU acceleration if necessary; PureML is available by default).
Big thanks to the upstream communities whose work makes this toolkit possible.
