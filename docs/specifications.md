# Specifications

## Scope and notation

| Symbol/status | Definition |
| --- | --- |
| PRE | Caller obligation; enforcement only where specified |
| POST | Successful-return guarantee |
| INV | Valid-state invariant |
| RAISES | Required failure behavior |
| LIMIT | Excluded guarantee or dependency limitation |
| V1 | Implemented: archive formats, reusable utilities, prerequisite fixes, schema validation, API freeze |
| V2 | Design draft: proposed interfaces; implementation/tests pending |

- Scope: V1 archives/reusable utilities; V2 extraction/publication.
- Exclusions: exhaustive V1 internals, legacy chunking/neural implementations, V2 learning.
- Enforcement: essential input/output/archive checks survive `python -O`.
- V2: `icontract` + shared predicates; tests before code; essential checks independent of decorator enablement.
- Remaining work: [V2_migration.md](../V2_migration.md).

## Frozen V1 public API

**INV:** preserve paths/exports/alias identity, constructors/public methods, signatures (names/order/kinds/defaults/annotations), return types/behavior.

| Module | Frozen surface |
| --- | --- |
| `sawnergy.rin` | `RINBuilder`, `build_rin`; `CpptrajScript`; `run_cpptraj` |
| `sawnergy.walks` | `Walker`, `walk`, `sample_walks`, resource lifetime; existing exports |
| `sawnergy.embedding` | `Embedder`, `embed_frame`, `embed_all`, `align_frames`; `Visualizer`, `build_frame`, `savefig`, `close` |
| `sawnergy.visual` | `Visualizer`, `build_frame`, `animate_trajectory`; color exports |
| `sawnergy.rin.rin_util` | Specified reusable CPPTRAJ interfaces |
| `sawnergy.sawnergy_util` | `ArrayStorage`; specified helpers; existing exports |
| `sawnergy.logging_util` | `configure_logging` path/signature |

- Baseline: `tests/test_public_api.py` + contracts/regressions below; legacy chunking/walk signatures included.
- Optional backend exports/dependencies preserved; PyTorch optional.
- Archives: version-1 layouts below; V2 requires separate interfaces/schema identities.
- Baseline includes intentional corrections: cleanup/error preservation; atomic compression; opening/read-only modes; write validation; script termination; empty batching; positive iteration steps; schema markers; rejection of incompatible/unmarked archives.
- Contract changes: explicit specifications/tests; no baseline replacement hiding regressions.
- Additions permitted; private/incidental names and V2 drafts excluded; package/producer releases independent of schemas.

## Existing archive contracts

### Common layout

**INV:** Zarr v3 root group; named arrays; JSON root attributes; working `.zarr`; published read-only `.zip`.

| Property | Contract |
| --- | --- |
| Block | `(K, *item_shape)`; append axis = 0 |
| Chunks | `(arrays_per_chunk, *[max(1, d) for d in item_shape])`; logical zero dimensions retained |
| Metadata maps | `array_chunk_size_in_block`: axis-0 chunk length; `array_shape_in_block`: item-shape list; `array_dtype_in_block`: `np.dtype(...).str`; keyed by block |
| Dataset discovery | Attribute -> block name; `None` = unavailable; missing = `KeyError`; zero-sized != unavailable |
| Compression round trip | Preserve immediate-array names/values/shapes/dtypes/chunks, root/array attributes |
| Responsibility | `ArrayStorage`: generic storage; producers/consumers: semantics, identities, units, schemas |

### Schema markers and compatibility

**Writer POST:** root markers `producer_version: str = sawnergy.__version__`, `schema_name: str`, `schema_version: int = 1`; no input-marker inheritance.

| Schema | Writer | Readers |
| --- | --- | --- |
| `rin` | `RINBuilder.build_rin` | `Walker.__init__`, `sawnergy.visual.Visualizer.__init__` |
| `rin_walks` | `Walker.sample_walks` | `Embedder.__init__` |
| `shallow_residue_embeddings` | `Embedder.embed_all` | `sawnergy.embedding.Visualizer.__init__` |

**Acceptance:** all markers present ∧ specified types ∧ nonboolean version ∧ expected `(schema_name,schema_version)`.

- Support: integer `1` per schema, independently versioned; no coercion/case normalization.
- `producer_version`: unparsed provenance; installed-version equality unnecessary.
- Package/channel/batch/walk/training changes alone do not change schema versions.
- Validator: `sawnergy._archive_schema.validate_archive_schema(root.attrs, *, schema_name, schema_version) -> None`.
- RAISES `ValueError`: missing/partial/malformed markers, wrong name, older/newer unsupported version. Diagnostic: offending keys or expected/encountered pair; unmarked = unsupported legacy.
- Order: validate BEFORE dataset-name lookup/reads/shared-memory allocation; rejection closes context, preserves bytes/attributes.
- Acceptance permits normal initialization/errors; dataset validity remains separately checked.
- Forbidden: inference, implicit versions, opening-time rewrite/automatic migration. Legacy support requires explicit migration/compatibility specification.
- `ArrayStorage`: no semantic marker generation/enforcement.

### RIN archives

**PRE:** residue count `N`; total source frames `F`; `1 <= start <= end <= F`; source frame IDs 1-based/inclusive.

**POST:** `B = frame_batch_size if > 0 else F`; `T = ceil((end-start+1)/B)` samples; final batch may be shorter.

| Name attribute | Default block | Shape | Meaning (`float32`) |
| --- | --- | --- | --- |
| `com_name` | `COM` | `(T,N,3)` | Batch-mean residue COM |
| `attractive_energies_name` | `ATTRACTIVE_energies` | `(T,N,N)` | Processed attractive magnitudes |
| `repulsive_energies_name` | `REPULSIVE_energies` | `(T,N,N)` | Processed repulsive magnitudes |
| `attractive_transitions_name` | `ATTRACTIVE_transitions` | `(T,N,N)` | Row-normalized attractive magnitudes |
| `repulsive_transitions_name` | `REPULSIVE_transitions` | `(T,N,N)` | Row-normalized repulsive magnitudes |

**Processing order:**

1. `A = mean_atomic_EMAP + mean_atomic_VMAP` over batch.
2. `E = P.T @ A @ P`; `P[a,r] = 1` iff atom `a` belongs to residue `r`.
3. Channels: `max(-E,0)`, `max(E,0)`; zero entries below each row's `prune_low_energies_frac` quantile; diagonal := 0; `M := (M+M.T)/2`.
4. Optional magnitude storage; transitions `Q[i,j] = M[i,j]/max(sum_j M[i,j],1e-12)`; zero rows remain zero.

**INV (finite inputs):** `M` symmetric/nonnegative/zero-diagonal; `Q` nonnegative/zero-diagonal, possibly asymmetric. Row mass >= `1e-12` -> sum ≈ 1; smaller positive mass -> sum < 1.

- Five discovery attributes required; `com_name="COM"`; disabled/unretained := `None`; magnitude switch: `keep_prenormalized_energies`.
- Metadata: `time_created`, `molecule_of_interest`, `frame_range`, `frame_batch_size`, `prune_low_energies_frac`.
- `frame_range`: supplied list or `None`, not necessarily resolved range.
- Membership order: numeric residue/atom-ID sort; COM order: CPPTRAJ residue loop.
- Downstream IDs: 1-based local rows; topology identities/units/source-frame arrays absent.
- LIMIT: no raw/separate V2 feature contract.

### Walk archives

**Definitions:** input `(T,N,N)` transitions; `L=walk_length` steps; `W=walks_per_node`; `S=int(round(W*saw_frac))`; `R=int(W)-S`.

| Name attribute | Default block | Shape (`uint16`) |
| --- | --- | --- |
| `attractive_RWs_name` | `ATTRACTIVE_RWs` | `(T,N*R,L+1)` |
| `attractive_SAWs_name` | `ATTRACTIVE_SAWs` | `(T,N*S,L+1)` |
| `repulsive_RWs_name` | `REPULSIVE_RWs` | `(T,N*R,L+1)` |
| `repulsive_SAWs_name` | `REPULSIVE_SAWs` | `(T,N*S,L+1)` |

**INV:** 1-based local IDs; order: timestamp/node/repetition. Enabled channel -> RW+SAW arrays, including zero counts; disabled -> discovery `None`.

- `walks_layout="time_leading_3d"`; time-aware axis 0 = starting timestamp; intermediate timestamps absent.
- SAW avoidance may relax on zero remaining probability mass; uniqueness not guaranteed.
- Structural metadata: four discovery attributes, `num_RWs`, `num_SAWs`, `node_count`, `walk_length`, `walks_per_node`, `time_stamp_count`.
- Other metadata: `time_created`, `seed`, `rng_scheme="SeedSequence.spawn_per_batch_v1"`, `num_workers`, `in_parallel`, `batch_size_nodes`.
- LIMIT: `N<=65535` for lossless IDs (unchecked); reproducibility depends on seed/batching; no serial/parallel equality. Time-awareness/stickiness/source identities absent.

### Shallow embedding archives

**POST:** `frame_embeddings_name="FRAME_EMBEDDINGS"`; `(T,N,D)`, `float32`; source timestamp/node order; `embedding_kind ∈ {in,out,avg}`; `objective ∈ {sg,sgns}`.

**INV:** frame models after first warm-start from preceding input/output weights; no automatic `align_frames`.

| Metadata category | Keys |
| --- | --- |
| Layout | `frame_embeddings_name`, `time_stamp_count`, `node_count`, `embedding_dim` |
| Provenance | `source_WALKS_path`, `walk_length`, `num_RWs`, `num_SAWs`, four source walk discovery attributes |
| Training | `objective`, `model_base`, `embedding_kind`, `num_epochs`, `batch_size`, `window_size`, `alpha`, `negative_sampling`, `num_negative_samples`, `lr_step_per_batch`, `shuffle_data`, `device_hint`, `model_kwargs_repr`, `RIN_type`, `using` |
| Reproducibility/storage | `master_seed`, `per_frame_seeds`, `arrays_per_chunk`, `compression_level`, `created_at` |

- Defaults/limits: `device_hint=""` if unspecified; `model_kwargs_repr` non-executable; no biological/alignment guarantee; V2 learning excluded.

### RIN visualizer initialization

**PRE:** compatible RIN/datasets. **POST:** `show=False` -> scatter + two attached empty 3D edge collections, no empty-data autoscaling; assignment/clearing supported; frame limits from COMs; signatures/interpretation unchanged.

## Reusable CPPTRAJ contracts (`rin_util.py`)

### `CpptrajScript`

**INV:** ordered string tuple; frozen fields; composition preserves operands; constructor types unchecked.

| Operation | PRE | POST / RAISES |
| --- | --- | --- |
| `from_cmd(cmd)` | String | `commands == (cmd,)` |
| `+ other` (`__add__`) | String/script | Ordered append/concatenation; unsupported operand -> `NotImplemented` / operator `TypeError` |
| `> target` | Script, or filename with nonempty commands | Script concatenation, or append `" out <target>"` to final command; empty+filename -> `IndexError` |
| `render()` | Valid CPPTRAJ lines/arguments | Newline join + trailing newline; append `run` iff exact `"run"` absent; empty -> `"run\n"` |

**LIMIT:** no `quit`, syntax parsing, case/whitespace normalization, path/mask quoting.

| Preset | Required datasets/output |
| --- | --- |
| `PAIRWISE_STDOUT` | `run`; `printdata PW[EMAP] square2d noheader`; same for `PW[VMAP]` |
| `COM_STDOUT(mol_id)` | `run`; residue dataset-coordinate loop; `printdata COMX* COMY* COMZ*` |

**PRE:** matching upstream datasets/syntax. **LIMIT:** presets neither parse/validate output nor implement generic V2 extraction.

### `locate_cpptraj(explicit=None, verify=True) -> str`

- PRE: `explicit=None` or executable `Path` (`.resolve()`); not an environment directory.
- Search order: explicit, `CPPTRAJ`, PATH `cpptraj`, PATH `cpptraj.exe`, `$AMBERHOME/bin/cpptraj`, `$CONDA_PREFIX/bin/cpptraj`.
- Deduplicate equal Paths before resolution, preserving order; aliases may repeat; invalid explicit permits fallback.
- POST: resolved absolute string of first existing/executable candidate; if verifying, `cpptraj -h` return code ∈ `{0,1}`.
- RAISES: exhausted candidates -> `CpptrajNotFound(RuntimeError)`, listing candidates; other launch errors propagate.
- LIMIT: 15-second timeout -> skip (docstring differs); no feature/numerical-compatibility guarantee.

### `run_cpptraj(cpptraj, script=None, argv=None, timeout=None, *, env=None)`

**PRE:** executable; string argv; complete script lines (final newline optional); timeout positive/`None`; script+argv permitted. `env=None` inherits; string dictionary replaces environment.

**POST:** execute `[cpptraj]+(argv or [])`, no shell; capture text stdout/stderr; return stdout; inputs unchanged.

| Input | stdin |
| --- | --- |
| `script=None` | `None` |
| Empty script | `"quit\n"` |
| Last effective line = standalone `quit`, case-insensitive | Original script unchanged |
| Otherwise | `script + sep + "quit\n"`; `sep="\n"` iff nonempty and no final LF |

**Effective line:** strip `#` comment and surrounding whitespace; discard blanks. Filename/argument/comment suffix `quit` is not termination.

**RAISES:** nonzero -> `subprocess.CalledProcessError` with output; timeout/launch errors propagate. **LIMIT:** line-based only; no continued/quoted-block parsing or output/physics validation.

### `CpptrajMaskParser.hierarchize_molecular_composition(mol_compositions_file)`

- PRE: readable UTF-8 table; first-line `[Mnum]`, `[Rnum]`, `[AtNum]`; integer fields per nonblank row. Duplicate header labels: last wins.
- POST: `{molecule_id: {residue_id: set(atom_ids)}}`; IDs unchanged; blank rows ignored; duplicate rows collapse; dictionaries retain first-appearance order; sets unordered; header-only -> `{}`.
- RAISES: empty file -> `RuntimeError`; missing columns, short rows, nonintegers -> `ValueError`; file errors propagate.
- LIMIT: no sorting/renumbering, positivity/ownership/correspondence validation. V2 establishes identity invariants separately.
- Legacy `com_parser`: exactly seven whitespace fields; first four -> CSV; no numeric validation (docstring weaker than implementation).

## Storage contracts (`sawnergy_util.py`)

### `ArrayStorage(pth, mode)` and lifetime

**PRE:** `pth: str or Path`; mode ∈ `{r,r+,a,w,w-}`. **POST:** expose resolved `store_path`, `store`, group `root`, `mode`; exact `.zip` -> read-only ZIP; otherwise suffix := `.zarr`.

| Local mode | Contract |
| --- | --- |
| `r` | Existing group; no directory/attribute writes |
| `r+` | Existing group; updates allowed; no creation |
| `a` | Open/create group |
| `w` | Create group; replace contents |
| `w-` | Create group; fail on existing stored data |

- Writable: initialize missing maps only. Read-only: missing maps accepted, attributes unchanged; missing `get_attr` -> `KeyError`.
- INV: fixed item shape/dtype/chunk length per helper-written block. Direct root mutation bypasses invariants; no numeric/physical/identity/semantic validation.
- RAISES: invalid path/mode/writable ZIP -> `ValueError` before acquisition; missing ZIP/local `r/r+` -> `FileNotFoundError`; array root (`r/r+/a`) -> `TypeError`; opening/metadata errors propagate without fallback.
- Failure after acquisition -> close attempt; original error/interruption preserved against ordinary closing errors (logged/suppressed).
- `close()`: best effort; context exit preserves body failure; post-close behavior unspecified.
- LIMIT: nontransactional modes (`w` may clear before failure). Zarr 3.1.2: side-effecting `assert` in `GroupMetadata.from_dict` retains `node_type` under `-O`, breaking reopening; no error suppression/bypass.

### Blocks and attributes

| Operation | PRE | POST / RAISES |
| --- | --- | --- |
| `write(these_arrays, to_block_named, *, arrays_per_chunk=None)` | Writable; finite homogeneous batch/stack; fixed shape/dtype; supplied chunk positive integer | Ordered append `(K,*item_shape)`; old rows retained; `None`; empty -> no mutation/chunk validation |
| `read(from_block_named, ids=None)` | Axis-0 array; integer/slice/1D integer indices | Detached ndarray, stored dtype; `None`: all; int: drop axis; slice/collection: retain axis/order; collections: NumPy-coerced orthogonal indices |
| `block_iter(from_block_named, *, step=1)` | Array; positive integer step | Ordered detached slices covering axis 0, length <= step; empty -> none; scalar -> one scalar ndarray |
| `list_blocks()` | Open store | Sorted immediate root-array names; no subgroup recursion |
| `delete_block(named)` | Writable existing member | Remove member and all three metadata entries |
| `add_attr(key,val)` | Writable; string key; JSON-compatible converted value | Set/replace root attribute |
| `get_attr(key)` | Existing attribute | Stored JSON-compatible value |

**RAISES:** read-mode mutation (even empty) -> `RuntimeError`; missing member/attribute -> `KeyError`; non-array read/iteration -> `TypeError`; selection/serialization errors propagate. `step<=0` -> `ValueError("step must be positive")` on iteration, even empty/scalar.

**Attribute conversion:** top-level only: NumPy scalar `.item()`, ndarray `.tolist()`, tuple/set -> list, date/datetime -> ISO. No recursive conversion/type restoration; unordered sets; maps remain consistent.

**Write validation order:** all item shapes/dtypes -> materialize exact `(len(these_arrays),*item_shape)` -> resolve cached settings -> validate existing array/type/axis/item shape/dtype/chunk -> create/fill metadata -> resize/assign.

| Write condition | Result |
| --- | --- |
| First chunk omitted | `10` + `RuntimeWarning`; later reuse cache |
| Resolved chunk <= 0 | `ValueError` |
| Within-batch shape/dtype mismatch; converted shape mismatch | `ValueError` |
| Cached shape/dtype/chunk conflict; existing chunk conflict | `RuntimeError` |
| Existing member type/axis/item-shape/dtype conflict | `TypeError` |
| Validation/conversion/promoted-warning failure | Contents/shapes/members/metadata unchanged; no partial metadata/new block |
| I/O/codec failure after validation | May leave partial creation/metadata/resize/assignment; no rollback |

### `compress(into=None, *, compression_level)`

- PRE (local): open store, writable destination, `int(level) ∈ [0,9]`; immediate arrays; copy memory = full source array.
- Destination: `None` -> source `.zip`; explicit case-insensitive `.zip` -> file; otherwise directory/source ZIP name. Create explicit parents.
- POST: preserve common round-trip fields; Blosc Zstd/shuffle at level (`0` still Blosc); return path string; source usable.
- Publication: temporary sibling ZIP -> copy -> successful close -> atomic replacement. Existing destinations replaced on success only.
- Pre-replacement failure: preserve destination bytes/absence; propagate creation/copy/close/replace/conversion errors; invalid range -> `ValueError`; attempt unlink even on interruption; ordinary removal errors logged, original preserved.
- ZIP source: return existing path; ignore destination/level, no validation/copy.
- LIMIT: no nested groups, power-loss durability, byte identity, codec/filesystem-metadata preservation, guaranteed cleanup.

### `ArrayStorage.compress_and_cleanup(output_pth, compression_level)`

**PRE:** compression contract. **POST:** yield temporary writable store; successful body -> compress -> close -> cleanup.

| Exit | Required behavior |
| --- | --- |
| Body failure/interruption | No compression/publication; destination unchanged; close/cleanup attempted; original failure |
| Compression failure | Close/cleanup attempted; original failure |
| Construction failure | No yield; directory cleanup attempted; original failure |
| Cleanup fails during active failure | Log ordinary cleanup error; preserve active failure |
| Cleanup fails after success | Propagate cleanup error |

**LIMIT:** `close()` remains best effort; cleanup attempted, not guaranteed.

## Other reusable helpers

| Interface | PRE | POST / RAISES / LIMIT |
| --- | --- | --- |
| `elementwise_processor(in_parallel=False, Executor=None, max_workers=None, capture_output=True)` | Finite input; compatible callable/args; parallel executor/workers; process picklability + main guard | Once/item; captured: ordered list (`[]` empty); uncaptured: await, return `None`; propagate errors |
| `batches_of(iterable, batch_size=-1, *, out_as=list, ranges=False, inclusive_end=False)` | Finite iterable; integer size; batch/pair wrapper | Consecutive nonempty batches; each item once/in order; positive size bounds length; nonpositive -> one batch; empty -> no yields/wrapper calls |
| `read_lines(file_path, skip_header=True)` | Readable UTF-8 | Retain newline characters with universal-newline conversion; optionally remove exactly first line; empty -> `[]`; file/decoding errors propagate |
| `temporary_file(prefix,suffix)` | Valid components; writable system temp directory | New empty file, closed handle, `Path`; caller owns deletion, including failures |
| `create_updated_subprocess_env(**var_vals)` | String names/convertible values; path-like returns string | Copy `os.environ`; `None`: remove; bool: `TRUE/FALSE`; path-like: `os.fspath`; else: `str`; parent/omitted unchanged |
| `compose_steps(*steps)` | `(callable,kwargs_or_None)` tuples; dict-convertible kwargs; compatible step values | Unary ordered composition; identity for zero steps; shallow kwargs copy per call; `None` output forwarded; errors propagate; input mutation allowed |
| `current_time()` | None | Local `%Y-%m-%d_%H%M%S`; no timezone/uniqueness/monotonicity guarantee |
| `is_main_process()` | None | No parent AND process name `MainProcess`; not proof of main guard |

- Processor: missing executor -> invocation-time `ValueError`; workers `None`/positive (falsy -> `os.cpu_count() or 1`); submit-all then collect; side-effect order/cancellation/rollback unspecified; caller owns shared state.
- Ranges: 0-based consumption `(start,stop_exclusive)`/`(start,end_inclusive)`, wrapped by `out_as`; flag ignored for data; not CPPTRAJ IDs. Slice sequences; materialize generic input, including ranges; nonpositive size unbounded-memory.
- Composition: tuple length/callability checked at construction (`TypeError`); kwargs conversion at invocation; nested mutable kwargs remain shared.

## V2 extraction contracts — design draft

### Feature decorator — implemented contract

`feature(*, kind: str, scope: str)` decorates subclass methods `(self, ctx)`.

**PRE:** `kind` in `{"pairwise", "global"}`; `scope` in `{"window", "trajectory"}`.

**RAISES:** invalid kind/scope -> `ValueError`, including under `python -O`.

**POST:** return the original method with `feature_kind == kind` and `feature_scope == scope`.
Do not execute or wrap the method; preserve signature, instance binding, return values, and exceptions.
No global registration. Shape/dtype/statistic/dependency contracts remain pending.

The remaining draft below predates the subclassing plan in `V2_migration.md` and awaits revision.

| Ownership | Proposed components |
| --- | --- |
| `feature_extractor.py` | Instantiable/configurable `FeatureExtractor`; `ExtractionContext` |
| `reprs.py` | `FeatureSpec`; `FrameRepresentation`; frame/trajectory contracts |
| Feature | Callable + specification; no required subclass/decorator hierarchy |

### Selection and context

**PRE:** nonempty `selection: molecule_id -> nonempty list[residue_id]`; positive nonboolean integers; distinct residues/molecule; valid topology membership.

**RAISES:** invalid types -> `TypeError`; unknown/wrong-molecule IDs -> `ValueError`.

**POST/INV:**

- `residue_ids`: immutable `(molecule_id,residue_id)` tuples, flattened mapping/list order.
- `atom_ids`: sorted complete atom tuples per residue, aligned with `residue_ids`; no discards/partial residues.
- Pairwise: include cross-molecule pairs; topology IDs != zero-based positions.
- Selection: resolve once, independently of outputs; new helper/method; parser API unchanged.

| `ExtractionContext` field | Contract |
| --- | --- |
| `cpptraj` | Resolved executable |
| `topology_file`, `trajectory_file` | Source paths |
| `windows` | Ordered batch tuple of source windows; each window nonempty/distinct original 1-based frame IDs within source range |
| `residue_ids`, `atom_ids` | Aligned immutable selection |
| `subprocess_env`, `timeout` | Forwarded child environment/limit |

**INV:** snapshotted/read-only context; fixed source/identities; consecutive requested-window batch slice; no omission/duplication/relabeling; identical frames across features/sample.

### Window statistics

**INV:** one output/window; singleton = original frame; statistic invariant to computational batching.

| Backend capability | Required action |
| --- | --- |
| Matching native window statistic | Use directly; exact window/denominator; no second reduction |
| Per-frame values | Accumulate/reduce in Python to declared statistic |
| Neither | `ValueError`; no substitute quantity |

- CPPTRAJ: [trajin](https://amberhub.chpc.utah.edu/trajin/) selects frames; [pairwise](https://amberhub.chpc.utah.edu/pairwise/) EMAP/VMAP averages processed frames; [vector center](https://amberhub.chpc.utah.edu/vector/) outputs COM trajectories. No universal averaging/per-window summaries; installed-backend reference tests required.
- Partial means: `mean = sum_b(n_b * mean_b)/sum_b(n_b)` using actual counts; other statistics require valid combination rules.
- Persist source frames/counts/statistic/weighting/normalization; missing frames -> error, unchanged denominator policy.
- Aggregation order fixed: generally `mean(distance(COM_r,COM_s)) != distance(mean(COM_r),mean(COM_s))`; mean H-bond count != fraction with any bond.
- No summary substitution for required per-frame dependencies; local intermediates permitted; cross-feature reuse requires explicit interface.
- No full-source-frame retention requirement; backend atomic-matrix memory unconstrained.

### Features and registration

```python
FeatureSpec(shape, dtype, units, statistic, requires=(), validate=None)
add_feature(self, name: str, extractor: FeatureCallable, specification: FeatureSpec) -> None
supported_features(self) -> tuple[str, ...]
```

| Field/state | INV |
| --- | --- |
| `shape` | Dimensions: nonnegative integers or `"N"`; `N` = selected residue count; >=1 residue axis; COM `("N",3)`; pairwise distance/energy `("N","N")` |
| `dtype` | Explicit NumPy real numeric/bool dtype; no implicit default |
| `units` | Nonempty string; dimensionless `"1"` |
| `statistic` | Explicit quantity/aggregation order |
| `requires` | Distinct, previously registered names; no missing/self dependency |
| Declarations/contracts | Snapshot identities/specifications; expectations invariant to caller mutation |
| Registry | Per instance; unique nonempty string names; callable extractors; valid specifications; registration order = execution/output order |
| Extraction | Registry fixed; >=1 feature; every registered dependency also stored; no implicit/hidden features |

**INV:** resolved shape/dtype/units/statistic constant per feature/trajectory; all residue axes share selection order.

**POST:** ordered registered-name tuple; failure leaves registry unchanged; no cross-instance/class mutation.

**RAISES `ValueError`:** duplicate names, missing/self dependencies. Proposed instance `add_feature` supersedes class-level `new_feature` stub.

### Feature callable and validator

```python
extract_feature(context: ExtractionContext, dependencies: Mapping[str, np.ndarray]) -> np.ndarray
validate(values: np.ndarray, dependencies: Mapping[str, np.ndarray]) -> None
```

**Callable PRE:** independent valid context; `set(dependencies)=set(requires)`; dependencies validated for current batch/windows/residue order and their declared statistics; borrowed/read-only, no retained mutation.

**Callable POST:** `B=len(context.windows)`; ndarray `(B,*resolved_shape)`; exact dtype; finite; `result[k]=statistic(window[k])`; axes follow `context.residue_ids`. Ownership -> extractor; no later mutation, implicit conversion/padding/missing values/aggregation.

**Validation:** generic type/shape/dtype/finiteness -> per-sample validator -> write; validator gets exact same-window dependencies; no mutation; success -> `None`.

**RAISES:** type/dtype -> `TypeError`; shape/nonfinite -> `ValueError`; backend/parser/custom errors propagate/abort. Callable cleans owned temporary resources.

**Backend:** validate/map source frame/entity/run-local IDs; row counts insufficient; semantic identity requires independent reference tests.

### Frame representation and contract

```python
FrameRepresentation(source_frame_ids, residue_ids, features)
FrameContract(residue_ids, features)  # Mapping[str, FeatureSpec]
FrameContract.validate(self, frame: FrameRepresentation) -> None
```

**INV:** one window/sample; source tuple labels original frames/aggregates; features -> sample arrays; units/statistics in contract; expectations from independent selection/registration, never checked arrays.

**Validation POST:** source IDs nonempty/distinct/positive integers; residue tuple = expected tuple; feature names = expected set; valid shapes/dtypes/finiteness/feature predicates.

**LIMIT:** position/window equality deferred to extractor/trajectory contract; borrowed slices permitted; frozen records do not imply immutable/correct arrays. No mutation between validation/write; post-caller-mutation validity unspecified.

### Trajectory contract

```python
TrajectoryContract(windows, frame_contract)
TrajectoryContract.validate(self, storage: ArrayStorage, *, batch_size: int = 1) -> None
```

**Representation:** `ArrayStorage` arrays/metadata; no full-RAM class.

**PRE:** nonempty windows, source-frame bounds validated before extraction; positive integer validation batch size.

**Validation POST:**

- Stored windows/counts/order = request; no missing/extra/duplicated/relabeled samples.
- Stored residue identities and feature names/shapes/dtypes/units/statistics/dependencies = contract.
- Every feature block shape `(S,*resolved_shape)`, `S=len(windows)`; equal channel lengths.
- Every stored sample satisfies frame/dependency predicates; metadata/structure before values; bounded reads; return `None`; mismatch raises.
- Validators: application code only; no archive-directed execution/imports.

### Extraction lifecycle

1. Validate registration, windows, selection; construct independent contracts.
2. Enter `ArrayStorage.compress_and_cleanup`; execute window batches/features in order.
3. Validate sample/window/frame contracts; append values and aligned identity/unit metadata incrementally.
4. Validate completed stored trajectory in bounded reads, including actual stored values.
5. Normal context exit -> publish; extraction/validation failure -> raise inside context, skip publication, attempt cleanup.

**Memory:** release completed batches/dependencies; batch size controls working data, windows/statistics aggregation.

### Unresolved V2 specifications

- Window construction, stride, overlap, final partial-window policy.
- Exact archive dataset/metadata keys and schema markers before trajectory-validator implementation.
- Built-in units/dtypes, tolerances, physical energy definitions/diagonals, missing-value policy.
- COM imaging/unwrapping and batch-boundary state; distance = Euclidean per-frame COM distance under agreed V1 coordinate processing, with explicit window aggregation order.
- Separate signed electrostatic/vdW features. Hydrogen-bond definitions/extraction: future work.

## Verification obligations and limits

| Suite | Contract coverage |
| --- | --- |
| `test_public_api.py` | Fixed signatures, exports/alias identity, distribution-derived version |
| `test_archive_schemas.py` | Reopened writer markers; real local/ZIP reader rejection before data/allocation; closure/unchanged bytes; different/non-release producers; no generic storage markers; optimized constructor checks via storage stub |
| `test_storage.py` | Write/append values/metadata/empty dimensions; read selection/copy/errors; attribute conversion/persistence; deletion/listing; modes/read-only behavior; failed initialization, validation/conversion/promoted warnings; publication/destination preservation/temporary cleanup/error precedence |
| `test_rin.py` | Script composition/rendering; mask IDs/order/duplicates/errors; executable priority/dedup/timeout/bypass/exhaustion; subprocess forwarding/termination/errors; finite pairwise output; channels/metadata/frame selection/partial COM means/atom sums/normalization floor |
| `test_sawnergy_util.py` | Empty/nonempty batching/ranges/wrappers; sequential/thread ordering/completion/arguments/errors/defaults; environment isolation; files/temporary ownership; composition/sharing/failures; clocks/process helpers |
| Walk/embedding/visual suites | Every node/frame transition; ordered paths/counts/IDs/empty channels/time-aware layout; embedding values/seeds/warm starts; plotting data flow; real empty/populated/cleared edge rendering |
| Required V2 tests before implementation | Instance isolation; failed-registration state; dependencies; window/batch identity; frame/trajectory mismatches; validation blocks publication; independent physical references; native/Python reduction equivalence; unequal partial counts; unsupported statistics/summary dependencies |

**LIMITS:** coverage incomplete; CPPTRAJ/shared-memory pipelines mocked; most GUI setup replaced; real renderer test retained. No native-extraction/process-worker, biological, power-loss, or guaranteed-cleanup proof. Local Torch validation unavailable with missing `libtorch_cpu.dylib`; stub archives do not validate Torch. Optimized storage-stub tests do not resolve Zarr 3.1.2's real-reopening failure.
