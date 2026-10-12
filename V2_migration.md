# To-dos:

*V1 work:*

- [x] Add package version to the top-level scope: `__version__ = version("sawnergy")` in `src/sawnergy/__init__.py`.
- [ ] Document existing archive contracts and public behavior to preserve, plus pre-conditions, post-conditions,
    and specifications for components reused by v2. Focus on the relevant parts of
    `src/sawnergy/rin/rin_util.py` and `src/sawnergy/sawnergy_util.py`; a full specification of all V1 internals
    is not required.
- [ ] Add archive attributes: `producer_version` (SAWNERGY version), `schema_name` (RIN, walks on a RIN,
    or shallow residue embeddings), and `schema_version` (format version, independent of package version).
    Update the writers in `RINBuilder.build_rin`, `Walker.sample_walks`, and `Embedder.embed_all`.
- [ ] Define the compatibility policy for existing archives without schema attributes.
- [ ] Reject incompatible archives before loading their datasets in `Walker.__init__`, `Embedder.__init__`,
    `visual.Visualizer.__init__`, and `embedding.Visualizer.__init__`. These checks must remain active under
    `python -O`. Keep `ArrayStorage` generic; share schema validation through a helper if useful.
- [ ] Specify and test storage failure behavior before reusing it in v2. In particular,
    `ArrayStorage.compress_and_cleanup` currently attempts to compress an archive even when its body raises.
- [ ] Add compatibility/regression tests for archive markers, supported and unsupported schemas, and unmarked archives.
- [ ] Freeze the documented existing API after these changes, preserving public import paths and behavior.

*V2 work:*

- [ ] Create `src/sawnergy/v2/` directory, therein will be:
    - `traj_features/`
        - `feature_extractor.py`
        - `reprs.py`
    - `embedding/`
        - `frame_ae.py`
        - `traj_ae.py`
- [ ] Include package `__init__.py` files and update `.gitignore` to include the new source and test directories.
- [ ] Specify extraction and representation contracts in `docs/specifications.md`: feature meanings, units,
    shapes, atom/residue identities and ordering, frame selection, missing values, and archive metadata/layout.
- [ ] Reuse suitable helpers from `rin_util.py` and `sawnergy_util.py` through imports. Use `ArrayStorage`
    for named arrays and metadata; v2 serialization and validation can live in `reprs.py` initially.
- [ ] Preserve separate physical features and explicit frame correspondence. The existing RIN pipeline sums
    electrostatic and van der Waals energies, transforms residue matrices, and averages over frame batches;
    `build_rin` is not a substitute for raw, per-frame v2 extraction.
- [ ] Design tests from the specifications in `tests/v2/`, then implement the components with `icontract`
    pre-conditions, post-conditions, and invariants. Keep essential input/archive validation always active.
- [ ] Complete extraction, representation, and archive I/O first (pipeline points 1 and 2); develop
    `frame_ae.py` and `traj_ae.py` afterward.
