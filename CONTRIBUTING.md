# Contributing to SAWNERGY

## Pull request target: `development` only

**All external pull requests must target `development`, strictly.** This is the
project's dev branch. Create your contribution branch from the current
`development` branch and select `development` as the **base branch** when opening
the PR on GitHub.

External PRs targeting `main` are not accepted; retarget them to `development`.
Maintainers handle promotion of reviewed and tested changes to `main`.

## Architecture

The existing V1 pipeline is:

```text
MD topology + trajectory → RINBuilder → RIN archive → Walker → Walks archive
                                                           → Embedder → Embedding archive
```

Each stage uses named arrays and metadata stored through `ArrayStorage`.
Working stores are Zarr directories; published archives are compressed ZIP files.

| Module / path | Responsibility |
| --- | --- |
| `src/sawnergy/rin/` | `RINBuilder` runs CPPTRAJ, aggregates atomic interaction maps into residue interactions, and writes attractive/repulsive energies, transition matrices, and COM snapshots. `rin_util.py` supplies executable discovery, script composition/execution, and output parsers. |
| `src/sawnergy/walks/` | `Walker` samples random and self-avoiding walks over RIN transition matrices, optionally moving between timestamps. `walker_util.py` provides shared-memory arrays and numerical helpers. |
| `src/sawnergy/embedding/` | `Embedder` builds skip-gram training pairs and trains SG/SGNS models using `SGNS_pml.py` (PureML) or `SGNS_torch.py` (optional PyTorch). The module also provides frame alignment and an embedding visualizer. |
| `src/sawnergy/visual/` | Renders and animates residue interaction networks using COM coordinates and energy channels; includes plotting and color utilities. |
| `src/sawnergy/sawnergy_util.py` | Shared Zarr array storage, archive compression/cleanup, batching, parallel processing, temporary files, environment handling, and pipeline composition. |
| `src/sawnergy/logging_util.py` | Configures console and rotating-file logging. |
| `src/sawnergy/v2/traj_features/` | V2 feature-extraction scaffolding. `feature_extractor.py` is intended to coordinate configurable residue features; `reprs.py` is intended to define representations and validation. These interfaces are still being specified. |
| `tests/` | Regression and numerical tests. `conftest.py` provides synthetic pipeline data and substitutes for external tools, shared memory, training, and plotting where appropriate. |
| [docs/specifications.md](docs/specifications.md) | Existing contracts and the proposed V2 extraction design. See also [V2_migration.md](V2_migration.md) for the migration plan. |

The V2 design uses explicit residue selections and source-frame windows. Each
feature declares its output shape, dtype, units, dependencies, and window
statistic. Its implementation uses a matching native CPPTRAJ summary or reduces
per-frame values in Python, rejecting unsupported configurations. This remains
planned behavior; the current stubs do not implement the complete workflow.

## Making changes

- Follow the surrounding code style and keep patches focused and small. Reuse
  existing helpers before introducing another abstraction.
- Preserve existing public import paths, signatures, and successful archive
  behavior. Explain any intentional contract or failure-behavior correction.
- The [frozen V1 API](docs/specifications.md#frozen-v1-public-api) and
  `tests/test_public_api.py` define the compatibility baseline for V2 work.
  Keep the existing behavior tests passing; do not change the signature baseline
  merely to silence a compatibility failure.
- For behavior changes, specify the expected contract, add targeted regression
  tests before implementation, then update `docs/specifications.md` to match.
- V2 implementation uses `icontract` for preconditions, postconditions, and
  invariants. Essential input/output and archive validation must remain active
  under `python -O`.
- Explain what changed, why, and how it was verified in the PR description.
  Include relevant test results, skips, and backend validation limits.

## CI test selection

Pushes and PRs targeting `main`, and release-tag workflows, run the full suite,
including the Torch job. Pushes and PRs targeting `development` select affected
test groups using `CICD/select_tests.py`, including downstream pipeline tests.
The Python 3.11/3.12 jobs exclude tests marked `torch` on selective runs; the
Torch job runs those tests only when the backend, embedding core, relevant tests,
or dependencies change. PureML backend tests are marked `pureml` separately.

Shared storage/RIN changes can affect downstream visualizers. Visualization
tests therefore still run when affected, even if no plotting source changed.
Mixed archive-schema tests are selected per reader/writer parameter. Public API
checks accompany source changes. Documentation-only changes run no tests; the
Python matrix reports successful no-op checks to retain required check names.
Workflow/selector, dependency, package initialization, or shared-fixture changes,
unmapped source/test files, and unavailable diff history trigger a full run.
If the selection job fails, both test jobs also fall back to the full suite.
Pushes compare the previous and new commit; PRs compare against the merge base.
Keep the dependency map current when adding components; new unclassified tests
are retained rather than silently deselected. With no selection environment
variable, local pytest continues to run all tests.

## Local setup and tests

Use Python 3.11 or newer in an isolated environment, then install the package
and test dependencies from the repository root:

```bash
python -m pip install -e ".[test]"
python -m pytest tests
```

Install `.[test,torch]` to exercise the optional PyTorch backend. CPPTRAJ is
required for real trajectory extraction; expose its executable through `PATH`,
`CPPTRAJ`, or an explicit executable path. Existing mocked CPPTRAJ tests do not
establish real backend correctness, so extraction changes should also be checked
against small, independently understood examples using CPPTRAJ.
