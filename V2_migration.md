# To-dos:

*V1 work:*

- [x] Add package version to the top-level scope: `__version__ = version("sawnergy")` in `src/sawnergy/__init__.py`.
- [x] Document existing archive contracts and public behavior to preserve, plus pre-conditions, post-conditions,
    and specifications for components reused by v2. Focus on the relevant parts of
    `src/sawnergy/rin/rin_util.py` and `src/sawnergy/sawnergy_util.py`; a full specification of all V1 internals
    is not required.
- [x] Add archive attributes: `producer_version` (SAWNERGY version), `schema_name` (RIN, walks on a RIN,
    or shallow residue embeddings), and `schema_version` (format version, independent of package version).
    Update the writers in `RINBuilder.build_rin`, `Walker.sample_walks`, and `Embedder.embed_all`.
- [x] Define the compatibility policy for existing archives without schema attributes.
    See `docs/specifications.md`: require all three typed markers and the reader's exact schema name/version;
    producer version is provenance only. Reject unmarked, partially marked, malformed, and unsupported archives.
- [x] Reject incompatible archives before loading their datasets in `Walker.__init__`, `Embedder.__init__`,
    `visual.Visualizer.__init__`, and `embedding.Visualizer.__init__`. These checks must remain active under
    `python -O`. Keep `ArrayStorage` generic; share schema validation through a helper if useful.
- [x] Specify and test storage failure behavior before reusing it in v2.
    Contracts are in `docs/specifications.md`; regression tests are in `tests/test_storage.py`.
    Body failures skip compression, cleanup preserves active errors, failed compression preserves
    existing destinations, and rejected writes leave storage unchanged. Storage I/O after write
    validation remains nontransactional, as documented.
- [x] Add compatibility/regression tests for archive markers, supported and unsupported schemas, and unmarked archives.
- [x] Freeze the documented existing API after these changes, preserving public import paths and behavior.
    See the frozen V1 API in `docs/specifications.md`, the signature/export baseline in
    `tests/test_public_api.py`, and the existing behavior regression tests.

*V2 work:*

- [x] Create `src/sawnergy/v2/` directory, therein will be:
    - `traj_features/`
        - `feature_extractor.py`
        - `reprs.py`
- [x] Include package `__init__.py` files and update `.gitignore` to include the new source and test directories.

### Feature extractor plan

1. **Subclass `FeatureExtractor`.** Mark feature methods with a global `@feature(...)` decorator;
   it records metadata, without a global feature registry or instance `.add_feature` API.
2. **Declare each feature's contract.** Include kind (`pairwise` or `global`), scope (`window` or
   `trajectory`), shape, dtype, units, statistic, and dependencies. All four kind/scope combinations
   are allowed. Pairwise outputs describe every selected residue pair, including cross-molecule pairs.
   Global outputs describe the selected system. COMs are not an initial output feature.
3. **Call `.featurize(...)`.** Supply trajectory, topology, CPPTRAJ executable path, selection mask,
   frame/window settings, and archive destination. The base class handles validation and archive creation.
4. **Resolve the mask.** Select complete residues across any number of molecules. Reject empty or
   partial-residue selections. Use topology residue order for every feature and save the identity mapping.
   All features compute using only the selected atoms, including global features.
5. **Run decorated methods in declaration order.** Each receives an explicit `ctx` containing input paths,
   selection, relevant frames/windows, and declared dependencies. Dependencies must appear earlier.
   Inherited features run first; decorated overrides retain their position and new features append.
6. **Validate and store results.** Methods return ndarrays. Window methods process batches; trajectory
   methods return one result. Validate results against their contracts, write incrementally with
   `ArrayStorage`, validate the completed archive, then compress and clean up.

Each feature implements its own statistic: use a matching CPPTRAJ summary, otherwise reduce per-frame
values in Python, otherwise reject the configuration. The base class does not average automatically.
Batch size is separate from window size. Preserve source-frame/window correspondence and keep physical
channels separate; V1 `build_rin` combines/transforms interactions and is not a V2 feature extractor.

**Proposed execution:** finish a feature over all selected frames before starting the next. This lets
trajectory features depend on completed window features. Read dependencies from storage in slices/batches
so the full trajectory's outputs need not fit in RAM.

**Still to confirm:** archive-backed dependency access; whether trajectory scope means the frames selected
by range/stride; and how distance-based masks establish fixed residue membership.

### Implementation order

- [ ] Finalize contracts in `docs/specifications.md`, replacing the earlier instance-registration design.
    Specify feature meanings, units, shapes, identities/order, frame/window selection, missing values,
    and archive metadata/layout.
- [ ] Write targeted contract tests in `tests/v2/` before implementation.
- [ ] Implement the decorator, context, and base `.featurize` with `icontract` conditions/invariants.
    Keep essential input/archive validation active under `python -O`.
- [ ] Reuse suitable `rin_util.py` and `sawnergy_util.py` helpers without changing V1 public APIs.
    Use `reprs.py` only if shared representation/validation code is needed.
- [ ] Complete incremental archive writing and failure cleanup; run the relevant tests.

Representation learning is outside the current scope.
