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
| V2 | Implemented foundations below; input preparation, extraction, archive format, and physical features planned |

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
| `traj_features` | V2 `.featurize()` — planned | V2 archive validation — planned |

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

- PRE (local): open store, writable destination, `int(level) ∈ [0,9]`; immediate arrays.
- Destination: `None` -> source `.zip`; explicit case-insensitive `.zip` -> file; otherwise directory/source ZIP name. Create explicit parents.
- POST: preserve common round-trip fields; Blosc Zstd/shuffle at level (`0` still Blosc); return path string; source usable.
- Copy one chunk-aligned multidimensional slice at a time, interleaving source reads and destination writes.
  Scalars copy once; zero-sized arrays retain metadata without data reads. Working data is bounded by one
  source chunk plus codec buffers. Each destination data chunk is written once.
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

## V2 extraction contracts

### Feature decorator — implemented contract

`feature(*, kind, scope, shape, dtype, units, statistic, requires=())` decorates subclass methods `(self, ctx)`.

**PRE:** `kind` in `{"pairwise", "global"}`; `scope` in `{"window", "trajectory"}`.
All output declarations are required.

| Declaration | Contract |
| --- | --- |
| `shape` | Per-sample tuple; pairwise exactly `("N", "N")`; global nonnegative nonboolean integer dimensions, including scalar `()`; `N = len(ctx.residue_ids)` |
| `dtype` | Explicit NumPy real numeric/bool dtype, normalized with `np.dtype`; exact output match |
| `units` | Nonblank string; dimensionless `"1"` |
| `statistic` | Nonblank string describing quantity and reduction order; implemented by feature method |
| `requires=()` | Tuple of distinct nonempty strings naming earlier feature attributes |

**POST:** return the original method with `feature_kind`, `feature_scope`, `feature_shape`, `feature_dtype`,
`feature_units`, `feature_statistic`, and `feature_requires` metadata. Preserve signature/binding/behavior.

**RAISES:** invalid dtype -> `TypeError`; invalid other declarations -> `ValueError` at decorator creation.
Missing required arguments -> `TypeError`. Checks remain active under `python -O`.

### Feature discovery

`FeatureExtractor._feature_methods(self) -> dict[str, MethodType]`

**POST:** fresh ordered mapping of feature names to methods bound to `self`.

- Inspect class dictionaries in reverse Python MRO, preserving declaration order within each class.
- Include ordinary instance functions carrying both `feature_kind` and `feature_scope`.
- Inherited features precede subclass additions. Decorated overrides retain their existing position.
- An undecorated override, including a non-method attribute, removes the inherited feature.
  Decorating that name again in a later subclass appends it as a new feature.
- Discovery evaluates class dictionaries only. Feature and property bodies remain unevaluated.
- Returned mappings belong to individual calls. Editing a returned mapping leaves later discovery intact.
- After resolving inheritance, every required name must identify an earlier feature in the final order.
  This also rejects cycles. Preserve discovery order.

**RAISES:** missing, self, or later dependencies -> `ValueError` identifying the dependent feature and
unavailable names, including under `python -O`. An override can remove or replace dependency declarations.

### Extraction context and dependency readers

`DependencyReader(storage: ArrayStorage, block_name: str)` borrows an open store and binds one block.

- Public reads delegate to `storage.read(block_name, ids)` through `read(ids=None)` and to
  `storage.block_iter(block_name, step=step)` through `block_iter(*, step=1)`.
- Indices select zero-based stored rows. Reads and batches return independent ndarray copies.
- Construction performs no dataset reads. Storage errors propagate through the read methods.
- Storage lifetime belongs to the caller. Reader access is valid while the borrowed store remains open.
- Invalid storage type -> `TypeError`; empty/non-string block name -> `ValueError`.

`ExtractionContext` is a frozen, keyword-only dataclass in `feature_extractor.py`.

| Field | Contract |
| --- | --- |
| `cpptraj`, `topology_file`, `trajectory_file` | `str`/`Path` inputs snapshotted as resolved `Path` objects |
| `selection_mask` | Read-only property derived from `residue_ids` as `:id1,id2,...`; topology-global residue IDs |
| `residue_ids` | Nonempty topology-ordered `(molecule_id, residue_id)` pairs; positive nonboolean integers; strictly increasing topology residue IDs |
| `atom_ids` | One nonempty, strictly increasing tuple of positive nonboolean atom IDs per residue; disjoint across residues |
| `frame_ids` | Nonempty, strictly increasing tuple of positive nonboolean 1-based source-frame IDs for this call |
| `windows=()` | Ordered source-frame tuples; each nonempty/strictly increasing; when present, their union equals `frame_ids`; overlaps allowed |
| `dependencies={}` | Snapshot mapping of nonempty feature names to `DependencyReader` objects |

**POST/INV:** nested identity/frame sequences are snapshotted as tuples; the dependency mapping is copied
and read-only. Field reassignment raises `FrozenInstanceError`; dependency-map mutation raises `TypeError`.
Structural value violations raise `ValueError`; non-iterable sequences and invalid path types raise
`TypeError`. Checks remain active under `python -O`.

**CALLER PRE:** source paths identify validated inputs; residue/atom membership matches the topology.
Dependencies are completed, validated outputs for exactly the current feature's `requires`.
Context construction validates structure and snapshots inputs. Input preparation validates sources and
topology membership. `.featurize()` will enforce dependency semantics and resource ownership.

### Feature outputs

`FeatureExtractor._validate_output(self, method, values, ctx) -> None`

**PRE:** discovered feature method and valid `ExtractionContext`; declarations fixed during extraction.
Feature methods accept `(self, ctx)` and return an ndarray. Dependencies are accessed through `ctx`.
Validation is explicit. The planned `.featurize()` calls this check before writing each output.

| Kind | Scope | Callable output shape | Archive block shape |
| --- | --- | --- | --- |
| Pairwise | Window | `(B, N, N)` | `(W, N, N)` |
| Global | Window | `(B, *shape)` | `(W, *shape)` |
| Pairwise | Trajectory | `(N, N)` | `(1, N, N)` |
| Global | Trajectory | `shape` | `(1, *shape)` |

`B = len(ctx.windows) > 0` for window calls; `W` is the complete trajectory's window count.
Trajectory features produce one result per call. Archive serialization is specified below.

**POST:** ndarray with exact declared dtype/shape; unmasked; every value finite; values unchanged;
return `None`. Scalars use zero-dimensional ndarrays. Explicit zero-sized global dimensions are allowed.

**RAISES:** non-ndarray/masked array/dtype mismatch -> `TypeError`; missing windows, shape mismatch,
or NaN/inf -> `ValueError`. Checks remain active under `python -O`.

**Residue ordering:** both pairwise axes index `ctx.residue_ids`, including cross-molecule pairs.
Feature implementations must validate/map backend atom/residue/frame IDs to the context before returning.
Reference tests must detect permutations. A bare ndarray carries no identity labels, so generic validation
checks axis sizes and cannot determine whether values were permuted. Symmetry/diagonals and physical
units/statistics require feature-specific specifications and independent numerical tests.

### Input preparation

`FeatureExtractor._prepare_inputs(topology_file, trajectory_file, cpptraj_path, selection, *,
frame_range=None, frame_stride=1, window_size=1) -> ExtractionContext`

- `selection` is a nonempty mapping `{molecule_id: residue_ids}`. Each value is a nonempty sequence
  of distinct positive nonboolean integer IDs. Residue IDs are CPPTRAJ topology-global IDs.
  Requested ordering is arbitrary; output follows topology residue order.
- Every requested molecule/residue pair must exist. Include each selected residue's complete atom set.
  Build the internal numeric residue mask from those validated identities.
- Source paths must be files; the explicit CPPTRAJ executable must be executable and verified through
  `locate_cpptraj`. Validate the topology/trajectory with native frame counting and a one-frame atom table.
  Reuse `run_cpptraj`, `CpptrajScript`, and `CpptrajMaskParser`.
- `frame_range` is a 1-based inclusive `(first, last)` tuple within the trajectory, defaulting to all frames.
  `frame_stride` and `window_size` are positive nonboolean integers. Selected frames are
  `range(first, last + 1, frame_stride)`. Trajectory features use all selected frames.
- Windows partition selected frames consecutively, with `window_size` frames per window. Keep the final
  short window and its actual frame IDs/count. Membership is independent of trajectory coordinates.
- The returned context contains all selected frames/windows and an empty dependency mapping.
  CPPTRAJ reads temporary safe-named links to support source filenames containing spaces.
  Links and atom-table files are cleaned up on success/failure; context paths retain the original sources.

`FeatureExtractor._window_batches(ctx, *, batch_size=1) -> Iterator[ExtractionContext]`

- Positive nonboolean `batch_size` counts windows. Require nonempty context windows.
- Yield consecutive window batches; each context contains their sorted source-frame union and the same
  sources, fixed residue/atom selection, and dependency readers. Batch size preserves windows and values.

**RAISES:** invalid selection container -> `TypeError`; empty selection, invalid/duplicate IDs, unknown or
wrong-molecule residues, invalid frame/window/batch parameters or out-of-bounds range -> `ValueError`;
missing/non-file source -> `FileNotFoundError`; non-executable CPPTRAJ -> `PermissionError`;
failed executable verification -> `CpptrajNotFound`; malformed frame-count output -> `RuntimeError`.
CPPTRAJ/parser errors propagate. Validation remains active under `python -O`.

### Execution — planned

- `.featurize(...)` accepts trajectory, topology, CPPTRAJ executable, fixed selection, frame/window settings,
  and destination. Require at least one discovered feature.
- Every feature computes from the selected atoms. Parser public APIs remain preserved.
- Validate source-frame bounds and requested windows independently of returned arrays. Source/identity
  order is fixed across features; batch boundaries preserve requested windows and sample correspondence.
- Proposed feature-major execution completes each feature before the next. Window methods process
  batches, trajectory methods run once, and contexts expose exactly their declared completed dependencies.
- Validate outputs before incremental `ArrayStorage` writes. Feature code owns its temporary resources.
  Backend/parser failures abort extraction with their original error.

### Statistics and physical features — planned

- Each feature implements its declared statistic with a matching native CPPTRAJ summary or Python
  reduction of per-frame values. Unsupported configurations raise `ValueError`.
- Window granularity and computational batch size are independent; batching preserves the statistic.
  Combine partial means with `sum(n_b * mean_b) / sum(n_b)` using actual frame counts.
- Persist source frames/counts/statistic/weighting/normalization. Missing frames are errors.
  Per-frame dependencies require per-frame information. Local intermediates remain feature-owned.
- Generally `mean(distance(COM_r, COM_s)) != distance(mean(COM_r), mean(COM_s))`.
  Mean H-bond count and the fraction of frames with any bond are distinct statistics.
- Native behavior requires installed-CPPTRAJ reference tests. Relevant actions include
  [trajin](https://amberhub.chpc.utah.edu/trajin/), [pairwise](https://amberhub.chpc.utah.edu/pairwise/),
  and [vector](https://amberhub.chpc.utah.edu/vector/).
- Initial physical methods cover COM distances and separate signed electrostatic/vdW interactions.
  COMs are internal intermediates; hydrogen bonds are future work.

### V2 archive format — version 1

**STATUS:** format specified; writer/reader implementation planned.

**INV:** common flat `ArrayStorage` layout. Published ZIPs contain immediate root arrays and JSON root
attributes. `N` residues, `A` selected atoms, `F` selected frames, `W` windows, and `K` features are positive.

| Required root attribute | Contract |
| --- | --- |
| `producer_version` | String `sawnergy.__version__`; provenance |
| `schema_name`, `schema_version` | `"traj_features"`, nonboolean integer `1`; common strict compatibility policy |
| `topology_file`, `trajectory_file`, `cpptraj` | Resolved original path strings used as provenance |
| `frame_range` | Original request as `null` or `[first, last]`, positive nonboolean integers, `first <= last` |
| `frame_stride`, `window_size` | Positive nonboolean integers; selected-frame stride and frames per window |
| `features` | Ordered nonempty JSON list of feature declarations below |
| `array_chunk_size_in_block`, `array_shape_in_block`, `array_dtype_in_block` | Existing `ArrayStorage` metadata maps covering exactly the stored arrays |

Additional attributes/manifest fields are permitted as inert JSON metadata.

| Required dataset | Shape | Values |
| --- | --- | --- |
| `residue_ids` | `(N, 2)` | Positive `(molecule_id, topology_residue_id)` pairs; strictly increasing residue IDs |
| `atom_ids` | `(A,)` | Positive topology atom IDs, concatenated in residue-row order; globally distinct |
| `atom_offsets` | `(N + 1,)` | Strictly increasing, first `0`, last `A`; residue `i` owns `atom_ids[offsets[i]:offsets[i+1]]`, strictly increasing within each residue |
| `frame_ids` | `(F,)` | Actual selected 1-based source-frame IDs, strictly increasing with difference `frame_stride` |
| `window_offsets` | `(W + 1,)` | Strictly increasing, first `0`, last `F`; window `w` uses `frame_ids[offsets[w]:offsets[w+1]]` |
| `feature_0` ... `feature_{K-1}` | Scope-dependent shapes above | Finite real numeric/bool values in manifest/residue/window order |

All five identity/offset datasets use logical dtype `int64`. Offsets index zero-based array positions.
Window lengths equal `window_size`, except the final length in `[1, window_size]`.
`W = ceil(F / window_size)`; window `w` corresponds to feature row `w` for every window-scope feature.
Trajectory features use all `frame_ids` and occupy row `0`. Stored trajectory scalars have shape `(1,)`.

Each `features[i]` object has these required fields.

| Field | Contract |
| --- | --- |
| `name`, `block` | Exact discovered method attribute name, unique/nonempty; `block == "feature_" + str(i)` |
| `kind`, `scope` | Declared `pairwise`/`global` and `window`/`trajectory` |
| `shape` | Pairwise `["N", "N"]`; global declared integer dimensions, scalar `[]` |
| `dtype` | Canonical numeric/bool type name `np.dtype(declared_dtype).name`, e.g. `"float32"` |
| `units`, `statistic` | Exact declared nonblank strings |
| `requires` | Exact declared dependency-name list; distinct names identifying earlier manifest entries |

Validate callable dtype exactly, then serialize its native-endian equivalent with lossless byte-order
normalization. Archived dtype comparisons use canonical numeric type names, including storage metadata
maps. Zarr encodes byte order. Preserve numeric type and values during serialization.
Shape/dtype/units/statistic remain fixed per feature. Explicit zero-sized global dimensions retain their
declared shapes and normal positive sample counts. Declarations are inert JSON. Validation uses application
code. A dependency reader returns stored sample rows; `read(0)` removes the sample axis.

**VALIDATION/PUBLICATION — planned:**

- Validate schema markers before dataset lookup, then root fields/manifest and exact required dataset set.
  Validate storage metadata against actual item shapes, logical dtypes, and chunk lengths.
- Validate identity/offset values, counts, complete window coverage, feature shapes/dtypes, and every stored
  value in bounded reads. For nonnull `frame_range`, `frame_ids == range(first, last+1, frame_stride)`.
  For `null`, frames start at `1`; final source coverage is checked against the prepared input context.
- Before publication, compare identities, frames, windows, declarations, and order with the independently
  prepared request. Feature implementations/reference tests establish physical meanings and axis alignment.
- Malformed structure/metadata/identities or missing/extra datasets -> `ValueError`; unsupported/mismatched
  dtype -> `TypeError`; nonfinite values -> `ValueError`. Checks survive `python -O`.
- Use `ArrayStorage.compress_and_cleanup`; completed validation precedes publication. Extraction/validation
  failures propagate inside the context. Keep outputs archive-backed and release working batches.
  Compression retains its documented full-array memory limit until streaming copy is implemented.

**Unresolved:** concrete physical units/dtypes/tolerances,
energy definitions/diagonals, COM imaging/unwrapping and batch-boundary state; subprocess environment/timeout.

## Verification obligations and limits

| Suite | Contract coverage |
| --- | --- |
| `test_public_api.py` | Fixed signatures, exports/alias identity, distribution-derived version |
| `test_archive_schemas.py` | Reopened writer markers; real local/ZIP reader rejection before data/allocation; closure/unchanged bytes; different/non-release producers; no generic storage markers; optimized constructor checks via storage stub |
| `test_storage.py` | Write/append values/metadata/empty dimensions; read selection/copy/errors; attribute conversion/persistence; deletion/listing; modes/read-only behavior; failed initialization, validation/conversion/promoted warnings; publication/destination preservation/temporary cleanup/error precedence |
| `test_rin.py` | Script composition/rendering; mask IDs/order/duplicates/errors; executable priority/dedup/timeout/bypass/exhaustion; subprocess forwarding/termination/errors; finite pairwise output; channels/metadata/frame selection/partial COM means/atom sums/normalization floor |
| `test_sawnergy_util.py` | Empty/nonempty batching/ranges/wrappers; sequential/thread ordering/completion/arguments/errors/defaults; environment isolation; files/temporary ownership; composition/sharing/failures; clocks/process helpers |
| Walk/embedding/visual suites | Every node/frame transition; ordered paths/counts/IDs/empty channels/time-aware layout; embedding values/seeds/warm starts; plotting data flow; real empty/populated/cleared edge rendering |
| `tests/v2/` | Decorator declarations/identity; discovery/order/overrides/dependencies; frozen context/IDs/snapshots; fixed selection/source/frame/window/batch preparation; native input preparation with spaced filenames; local/ZIP dependency reads; output shapes/dtypes/finiteness/masks; optimized validation |
| Remaining V2 tests | Backend identity/order alignment; window/batch correspondence; validation blocks publication; independent physical references; native/Python reduction equivalence; unequal partial counts; unsupported statistics/summary dependencies; full archive validation |

**LIMITS:** coverage incomplete; physical CPPTRAJ/shared-memory pipelines mocked; native input preparation
and real renderer tests retained. Native input tests require `CPPTRAJ` or an executable on PATH.
Physical extraction/process-worker, biological, power-loss, and guaranteed-cleanup validation remain pending.
Local Torch validation unavailable with missing `libtorch_cpu.dylib`; stub archives do not validate Torch.
Optimized storage-stub tests do not resolve Zarr 3.1.2's real-reopening failure.
