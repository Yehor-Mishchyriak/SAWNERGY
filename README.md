# Sawnergy

A modular pipeline for transforming molecular dynamics (MD) trajectories into rich graph representations, sampling
random and self-avoiding walks, learning node embeddings, and visualising residue interaction networks (RINs). Sawnergy
keeps the full workflow — from `cpptraj` output to skip-gram embeddings — inside Python, backed by efficient Zarr-based
archives and optional GPU acceleration.

---

## Why Sawnergy?

- **Bridge simulations and graph ML**: Convert raw MD trajectories into residue interaction networks ready for graph
  algorithms and downstream machine learning tasks.
- **Deterministic, shareable artefacts**: Every stage produces compressed Zarr archives with explicit metadata so runs
  can be reproduced, shared, or inspected later.
- **High-performance data handling**: Heavy arrays live in shared memory during walk sampling; archives are written in
  chunked, compressed form for fast read/write.
- **Flexible embedding backends**: Train skip-gram with negative sampling (SGNS) models using either PureML or PyTorch
  with identical APIs.
- **Visualization out of the box**: Plot and animate residue networks without leaving Python, using the same artefacts
  produced earlier in the pipeline.

---

## Pipeline at a Glance

```
MD Trajectory + Topology
          │
          ▼
   RINBuilder (cpptraj)
          │   →  RIN archive (.zip/.zarr)
          ▼
     Walker (SharedNDArray)
          │   →  Walks archive (RW/SAW per frame)
          ▼
  Embedder (SGNS backends)
          │   →  Embedding archive (frame × vocab × dim)
          ▼
   Visualizer / Downstream ML
```

Each stage consumes the archive produced by the previous one. Metadata embedded in the archives ensures frame order,
node indexing, and RNG seeds stay consistent across the toolchain.

---

## Core Components

### `sawnergy.rin.RINBuilder`

* Wraps the AmberTools `cpptraj` executable to:
  - compute per-frame electrostatic (EMAP) and van der Waals (VMAP) energy matrices at the atomic level,
  - project atom–atom interactions to residue–residue interactions using compositional masks,
  - prune, symmetrise, remove self-interactions, and L1-normalise the matrices,
  - compute per-residue centres of mass (COM) over the same frames.
* Outputs a compressed Zarr archive with transition matrices, optional prenormalised energies, COM snapshots, and rich
  metadata (frame range, pruning quantile, molecule ID, etc.).
* Supports parallel `cpptraj` execution, batch processing, and keeps temporary stores tidy via
  `ArrayStorage.compress_and_cleanup`.

### `sawnergy.walks.Walker`

* Attaches to the RIN archive and loads attractive/repulsive transition matrices into shared memory using
  `walker_util.SharedNDArray` so multiple processes can sample without copying.
* Samples random walks (RW) and self-avoiding walks (SAW), optionally time-aware, with reproducible seeding derived from
  the RIN’s master seed.
* Persists walks as `(time, walk_id, length+1)` tensors (1-based node indices) alongside metadata such as
  `walk_length`, `walks_per_node`, and RNG scheme.

### `sawnergy.embedding.Embedder`

* Consumes walk archives, generates skip-gram pairs, and normalises them to 0-based indices.
* Provides a unified interface to SGNS implementations:
  - **PureML backend** (`SGNS_PureML`): works with the `pureml` ecosystem, optimistic for CPU training.
  - **PyTorch backend** (`SGNS_Torch`): uses `torch.nn.Embedding` with Adam optimiser; plays nicely with GPUs.
* Exposes `embedd_frame` (single frame) and `embedd_all` (all frames, deterministic seeding per frame) which return the
  learned input embedding matrices and write them to disk when requested.

### `sawnergy.visual.Visualizer`

* Opens RIN archives, resolves dataset names from attributes, and renders nodes plus attractive/repulsive edge bundles
  in 3D using Matplotlib.
* Handles backend selection (`Agg` fallback in headless environments) and offers convenient colour palettes via
  `visualizer_util`.

### Supporting Utilities

* `sawnergy.sawnergy_util`
  - `ArrayStorage`: thin wrapper over Zarr v3 with helpers for chunk management, attribute coercion, and transparent
    compression to `.zip` archives.
  - Parallel helpers (`elementwise_processor`, `compose_steps`, etc.), temporary file management, logging, and runtime
    inspection utilities.
* `sawnergy.logging_util.configure_logging`: configure rotating file/console logging consistently across scripts.

---

## Archive Layouts

| Archive                | Key datasets                                            | Important attributes                                                                 |
|------------------------|---------------------------------------------------------|---------------------------------------------------------------------------------------|
| **RIN**                | `ATTRACTIVE_transitions`, `ATTRACTIVE_energies`, `COM`  | `frame_range`, `node_count`, `time_stamp_count`, `prune_low_energies_frac`, RNG seed  |
| **Walks**              | `ATTRACTIVE_RWs`, `ATTRACTIVE_SAWs` (optional)          | `walk_length`, `walks_per_node`, `num_RWs`, `num_SAWs`, `SeedSequence.spawn_per_batch`|
| **Embeddings**         | `FRAME_EMBEDDINGS`                                      | `model_base`, `dimensionality`, `frames_written`, `rng_scheme`, training hyperparams  |

All archives are Zarr groups wrapped into `.zip` files by default, making them easy to distribute and inspect with
standard tools.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-org>/sawnergy.git
   cd sawnergy
   ```
2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install core dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Optional backends**
   - PureML backend: `pip install ym-pure-ml`
   - PyTorch backend (choose build for your platform/GPU): follow [https://pytorch.org/get-started](https://pytorch.org/get-started)

> **Note:** RIN building requires `cpptraj` (AmberTools). Ensure it is discoverable via `$PATH` or the `CPPTRAJ`
> environment variable.

---

## Quick Start

```python
from pathlib import Path
from sawnergy.logging_util import configure_logging
from sawnergy.rin import RINBuilder
from sawnergy.walks import Walker
from sawnergy.embedding import Embedder

configure_logging("./logs")

# 1. Build a Residue Interaction Network archive
rin_path = Path("./RIN_demo.zip")
rin_builder = RINBuilder()
rin_builder.build_rin(
    topology_file="system.prmtop",
    trajectory_file="trajectory.nc",
    molecule_of_interest=1,
    frame_range=(1, 100),
    frame_batch_size=10,
    prune_low_energies_frac=0.3,
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

# 3. Train embeddings per frame
embedder = Embedder(walks_path, base="torch", seed=999)
embeddings_path = embedder.embedd_all(
    RIN_type="attr",
    using="merged",
    window_size=4,
    num_negative_samples=5,
    num_epochs=5,
    batch_size=1024,
    dimensionality=128,
    shuffle_data=True,
    output_path="./EMBEDDINGS_demo.zip",
    sgns_kwargs={"device": "cuda"},
)
print("Embeddings written to", embeddings_path)
```

---

## Visualisation

```python
from sawnergy.visual import Visualizer

viz = Visualizer("./RIN_demo.zip", show=False)
viz.update_frame(0)            # first frame
viz.highlight_nodes([1, 10])   # optional helpers
viz.save("frame0.png")
```

`Visualizer` lazily loads datasets and works even in headless environments (falls back to the `Agg` backend).

---

## Advanced Notes

- **Time-aware walks**: Set `time_aware=True`, provide `stickiness` and `on_no_options` when calling `Walker.sample_walks`.
- **Shared memory lifecycle**: Always call `Walker.close()` (or use a context manager) to release shared-memory segments.
- **PureML vs PyTorch**: The `Embedder` factory automatically imports the requested backend; helpful error messages guide
  installation when a backend is missing.
- **ArrayStorage utilities**: Use `ArrayStorage` directly to peek into archives, append arrays, or manage metadata.

---

## Testing & Quality Assurance

The automated test suite (`pytest`) synthesises deterministic cpptraj outputs and exercises the entire workflow:

- RIN parsing, residue aggregation, and metadata verification.
- Random/self-avoiding walk sampling and probability consistency with the RIN.
- Embedding orchestration, frame ordering, SGNS pair generation property tests.
- PureML and PyTorch SGNS smoke tests verifying finite weights and decreasing loss.
- Visualiser smoke tests that cover data loading and artist updates.

Run the suite (inside the project virtualenv):

```bash
python -m pytest
```

Optional extras (`pureml`, `torch`) are skipped automatically if not installed.

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
├── tests/             # Synthetic end-to-end tests (pytest)
├── requirements.txt   # Core dependency pinning
└── README.md
```

---

## Contributing

1. Fork and clone the repository.
2. Create a feature branch off `main`.
3. Make your changes, add unit tests where appropriate.
4. Run `python -m pytest` (install optional dependencies for full coverage).
5. Submit a pull request with a clear description.

Feel free to open issues for feature requests, questions, or bug reports.

---

## Acknowledgements

Sawnergy builds on the AmberTools `cpptraj` ecosystem, NumPy/SciPy tooling, Zarr for storage, PureML, and PyTorch. Big
thanks to the upstream communities whose work makes this toolkit possible.

