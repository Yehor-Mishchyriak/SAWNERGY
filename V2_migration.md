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

Keep correctness and conciseness equally important. Specify contracts, write targeted tests, then implement
the smallest clear change. Reuse existing helpers and keep essential validation active under `python -O`.

### Implemented

- [x] V2 source/test packages and `.gitignore` entries.
- [x] Subclass-based `FeatureExtractor` with metadata-only `@feature`.
- [x] Ordered discovery, inherited features, and decorated/undecorated overrides.
- [x] Explicit `requires=()` with earlier-dependency validation after inheritance.
- [x] Frozen `ExtractionContext` and read-only `DependencyReader` using `ArrayStorage`.

### Remaining implementation order

1. [x] **Feature output contracts.** Require shape, dtype, units, and statistic in `@feature`.
   Validate ndarray outputs, scope-dependent shapes, exact dtype, and finite/unmasked values.
   Feature implementations must align backend residue IDs with `ctx.residue_ids`, verified by tests.
   Replace the older specification sections with the current subclass-based contracts.
2. [x] **Input preparation.** Accept `selection={molecule_id: [residue_ids...]}` and validate every pair.
   Derive the CPPTRAJ mask from complete residues/atoms in topology order. Validate sources and CPPTRAJ.
   Use a 1-based inclusive frame range and stride. Trajectory scope uses all selected frames.
   Build nonoverlapping windows of selected frames, keep the final short window, and batch by window count.
   All feature computation uses this fixed selection.
3. [x] **V2 archive format.** Specify schema markers, feature blocks, residue/atom identities,
   frame/window metadata, and feature declarations.
   See `docs/specifications.md` for `traj_features` schema version 1, flat ID/offset datasets,
   numbered feature blocks, an ordered feature manifest, and a sample axis for every feature block.
4. [ ] **`.featurize()`.** Accept sources, CPPTRAJ path, fixed selection, frame/window settings, and destination.
   Prepare contexts, run features in declaration order, expose completed dependencies, validate/write
   incrementally, validate the finished archive, then compress and clean up through `ArrayStorage`.
   Proposed execution completes each feature before starting the next, supporting trajectory features
   that consume window outputs through archive readers.
   Address the existing full-array compression copy to preserve bounded-memory publication.
5. [ ] **Concrete feature methods.** Implement distances, separate signed electrostatic interactions,
   and vdW interactions after specifying their physical definitions, units, and reduction rules.
   COMs may be internal intermediates. Hydrogen bonds remain future work.
6. [ ] **Integration and numerical verification.** Check independent CPPTRAJ references, batching
   consistency, archive round trips, and failure cleanup. Run the relevant tests at every step.

Each feature implements its declared statistic using a matching CPPTRAJ summary or Python reduction,
rejecting unsupported configurations. Window granularity is independent of computational batch size.
Kind (`pairwise`/`global`) and scope (`window`/`trajectory`) are independent. Use `reprs.py` only if needed.
