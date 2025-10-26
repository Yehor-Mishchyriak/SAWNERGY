# SAWNERGY

[![PyPI](https://img.shields.io/pypi/v/sawnergy)](https://pypi.org/project/sawnergy/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/Yehor-Mishchyriak/SAWNERGY/blob/main/LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

A toolkit for transforming molecular dynamics (MD) trajectories into rich graph representations, sampling
random and self-avoiding walks, learning node embeddings, and visualizing residue interaction networks (RINs). SAWNERGY
keeps the full workflow — from `cpptraj` output to skip-gram embeddings (node2vec approach) — inside Python, backed by efficient Zarr-based archives and optional GPU acceleration.

---

## Installation

   ```bash
   pip install sawnergy
   ```

> **Optional:** For GPU training, install PyTorch separately (e.g., `pip install torch`).
> **Note:** RIN building requires `cpptraj` (AmberTools). Ensure it is discoverable via `$PATH` or the `CPPTRAJ`
> environment variable. Probably the easiest solution: install AmberTools via conda, activate the environment, and SAWNERGY will find cpptraj executable on its own, so just run your code and don't worry about it.

---

# UPDATES:

## v1.0.7 — What’s new:
- **Added plain SkipGram model**
  - Now, the user can choose if they want to apply the negative sampling technique (two binary classifiers) or train a single classifier over the vocabulary (full softmax). For more detail, see: [node2vec](https://arxiv.org/pdf/1607.00653), [word2vec](https://arxiv.org/pdf/1301.3781), and [negative_sampling](https://arxiv.org/pdf/1402.3722).
- **Set a harsher default for low interaction energies pruning during RIN construction**
  - Now we zero out 85% of the lowest interaction energies as opposed to the past 30% default, leading to more meaningful embeddings.
- **BUG FIX: Visualizer**
  - Previously, the visualizer would silently draw edges of 0 magnitude, meaning they were actually being drawn but were invisible due to full transparency and 0 width. As a result, the displayed image / animation would be very laggy. Now, this was fixed, and given high pruning default, the displayed interaction networks are clean and smooth under rotations, dragging, etc.
- **New Embedding Visualizer (3D)**
  - New lightweight viewer for per-frame embeddings that projects embeddings with PCA to a **3D** scatter. Supports the same node coloring semantics, optional node labels, and the same antialiasing/depthshade controls. Works in headless setups using the same backend guard and uses a blocking `show=True` for scripts.

---

## Why SAWNERGY?

- **Bridge simulations and graph ML**: Convert raw MD trajectories into residue interaction networks ready for graph
  algorithms and downstream machine learning tasks.
- **Deterministic, shareable artifacts**: Every stage produces compressed Zarr archives that contain both data and metadata so runs can be reproduced, shared, or inspected later.
- **High-performance data handling**: Heavy arrays live in shared memory during walk sampling to allow parallel processing without serialization overhead; archives are written in chunked, compressed form for fast read/write.
- **Flexible objectives & backends**: Train Skip-Gram with **negative sampling** (`objective="sgns"`) or **plain Skip-Gram** (`objective="sg"`), using either **PureML** (default) or **PyTorch**.
- **Visualization out of the box**: Plot and animate residue networks without leaving Python, using the data produced by RINBuilder

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

---

## Core Components

### `sawnergy.rin.RINBuilder`

* Wraps the AmberTools `cpptraj` executable to:
  - compute per-frame electrostatic (EMAP) and van der Waals (VMAP) energy matrices at the atomic level,
  - project atom–atom interactions to residue–residue interactions using compositional masks,
  - prune, symmetrize, remove self-interactions, and L1-normalise the matrices,
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

* Consumes walk archives, generates skip-gram pairs, and normalises them to 0-based indices.
* Provides a unified interface to SGNS implementations:
  - **PureML backend** (`SGNS_PureML`): works with the `pureml` ecosystem, optimistic for CPU training.
  - **PyTorch backend** (`SGNS_Torch`): uses `torch.nn.Embedding` plays nicely with GPUs.
* Both `SGNS_PureML` and `SGNS_Torch` accept training hyperparameters such as batch_size, LR, optimizer and LR_scheduler, etc.
* Exposes `embed_frame` (single frame) and `embed_all` (all frames, deterministic seeding per frame) which return the
  learned input embedding matrices and write them to disk when requested.

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
| **Embeddings** | `FRAME_EMBEDDINGS` → **(frames_written, vocab_size, D)**, typically float32 | `time_created` (ISO) • `seed` (int) • `rng_scheme` = `"SeedSequence.spawn_per_frame_v1"` • `source_walks_path` (str) • `model_base` = `"torch"` or `"pureml"` • `rin_type` = `"attr"` or `"repuls"` • `using_mode` = `"RW"|"SAW"|"merged"` • `window_size` (int) • `alpha` (float; noise exponent) • `dimensionality` = D • `num_negative_samples` (int) • `num_epochs` (int) • `batch_size` (int) • `shuffle_data` (bool) • `frames_written` (int) • `vocab_size` (int) • `frame_count` (int) • `embedding_dtype` (str) • `frame_embeddings_name` = `"FRAME_EMBEDDINGS"` • `arrays_per_chunk` (int) • `compression_level` (int) • `objective` = `"sgns"` or `"sg"` |

**Notes**

- In **RIN**, `T` equals the number of frame **batches** written (i.e., `frame_range` swept in steps of `frame_batch_size`). `ATTRACTIVE/REPULSIVE_energies` are **pre-normalised** absolute energies (written only when `keep_prenormalized_energies=True`), whereas `ATTRACTIVE/REPULSIVE_transitions` are the **row-wise L1-normalised** versions used for sampling.
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
    include_repulsive=False,
)

# 2. Sample walks from the RIN
walker = Walker(rin_path, seed=123)
walks_path = Path("./WALKS_demo.zip")
walker.sample_walks(
    walk_length=16,
    walks_per_node=32,
    saw_frac=0.25,
    include_attractive=True,
    include_repulsive=False,
    time_aware=False,
    output_path=walks_path,
    in_parallel=False,
)
walker.close()

# 3. Train embeddings per frame (PyTorch backend)
import torch

embedder = Embedder(walks_path, base="torch", seed=999)
embeddings_path = embedder.embed_all(
    RIN_type="attr",
    using="merged",
    window_size=4,
    objective="sgns",
    num_negative_samples=5,
    num_epochs=5,
    batch_size=1024,
    dimensionality=128,
    shuffle_data=True,
    output_path="./EMBEDDINGS_demo.zip",
    sgns_kwargs={
        "optim": torch.optim.Adam,
        "optim_kwargs": {"lr": 1e-3},
        "lr_sched": torch.optim.lr_scheduler.LambdaLR,
        "lr_sched_kwargs": {"lr_lambda": lambda _: 1.0},
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
)
print("Embeddings written to", embeddings_path)
```

> For the PureML backend, supply the relevant optimiser and scheduler via `sgns_kwargs`
> (for example `optim=pureml.optimizers.Adam`, `lr_sched=pureml.optimizers.CosineAnnealingLR`).

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

viz = sawnergy.embedding.Visualizer("./EMBEDDINGS_demo.zip")
viz.build_frame(1, show=True)
```

---

## Advanced Notes

- **Time-aware walks**: Set `time_aware=True`, provide `stickiness` and `on_no_options` when calling `Walker.sample_walks`.
- **Shared memory lifecycle**: Call `Walker.close()` (or use a context manager) to release shared-memory segments.
- **PureML vs PyTorch**: Choose the backend via `Embedder(..., base="pureml"|"torch")` and provide backend-specific
  constructor kwargs through `sgns_kwargs` (optimizer, scheduler, device).
- **ArrayStorage utilities**: Use `ArrayStorage` directly to peek into archives, append arrays, or manage metadata.

---

## Project Structure

```
├── sawnergy/
│   ├── rin/           # RINBuilder and cpptraj integration helpers
│   ├── walks/         # Walker class and shared-memory utilities
│   ├── embedding/     # Embedder + SGNS backends (PureML / PyTorch)
│   ├── visual/        # Visualizer and palette utilities
│   ├── logging_util.py
│   └── sawnergy_util.py
│
└── README.md
```

---

## Acknowledgements

SAWNERGY builds on the AmberTools `cpptraj` ecosystem, NumPy, Matplotlib, Zarr, and PyTorch (for GPU acceleration if necessary; PureML is available by default).
Big thanks to the upstream communities whose work makes this toolkit possible.
