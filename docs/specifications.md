# Specifications

## Scope and status

This document specifies the existing archive formats and reusable public behavior
needed for V2 feature extraction. The V1 baseline is based on
`src/sawnergy/rin/rin_util.py`, `src/sawnergy/sawnergy_util.py`, and the current
V1 archive writers/readers. The V2 design below specifies proposed interfaces
and structural contracts; it is not an implemented API or a complete new archive
format. It does not specify every V1 internal.

**Preconditions** are caller obligations, not claims that every obligation is
currently checked. **Postconditions** apply after successful return.
**Invariants** describe valid objects/data. Explicit checks and limitations are
identified separately. No `icontract` decorators are added by this document;
essential input/archive validation must remain active under `python -O` when
contracts are implemented. Known defects below are recorded for tests and later
fixes, not requirements to preserve them.

The prerequisite patch queue is complete. The policy for unmarked archives,
reader compatibility validation, and completion of the V2 extraction/archive
specification remain pending tasks in `V2_migration.md`.

### Compatibility boundary for prerequisite patches

Public import paths, signatures, parameter names/defaults, return types, archive
layouts, and successful non-boundary behavior are retained. The following
failure/boundary corrections are intentional compatibility exceptions:

- `compress_and_cleanup` skips publication after body failure, attempts cleanup
  on every exit, and preserves the active error if cleanup also fails.
- `compress` publishes only a fully closed temporary ZIP, preserving an existing
  destination on failure before replacement.
- Construction respects opening modes, avoids read-only attribute writes, and
  propagates opening errors instead of retrying with group creation.
- `write` rejects incompatible batches and chunk settings before mutation;
  conversion errors and promoted default-chunk warnings also leave storage intact.
- `run_cpptraj` recognizes a standalone terminal `quit` and appends a separate
  command when needed, including scripts without a final newline.
- `batches_of` yields nothing for every empty input and valid batch size, rather
  than raising for empty sequences or fabricating empty batches/ranges.
- `block_iter` rejects zero/negative steps, including on empty/scalar arrays.

The detailed contracts below define the precise limits of these corrections;
they do not promise general transactional writes or infallible filesystem cleanup.

## Existing archive contracts and public behavior to preserve

### Common layout

- V1 writes Zarr v3 root groups containing named arrays and JSON-compatible root
  attributes. Working stores are `.zarr` directories; final archives are `.zip`
  files opened read-only through `ArrayStorage`.
- Each block has shape `(K, *item_shape)`, with axis 0 used for appends. Successful
  compression/reopening preserves array names, values, shape, dtype, chunk shape,
  root attributes, and array attributes for immediate root arrays.
- Readers resolve dataset names through root attributes. A name value of `None`
  means unavailable; this differs from a missing key (`KeyError`) and from a
  present array with a zero-sized dimension.
- Three root mappings, keyed by block name, describe storage:
  `array_chunk_size_in_block` holds the axis-0 chunk length,
  `array_shape_in_block` the item shape as a list, and
  `array_dtype_in_block` the NumPy dtype string (`np.dtype(...).str`).
  Chunk shape is `(arrays_per_chunk, *[max(1, d) for d in item_shape])`;
  data dimensions may be zero although chunk dimensions must be positive.
- Writers add the root attributes `producer_version`, `schema_name`, and
  `schema_version` as specified below. Readers do not yet validate these markers;
  acceptance/rejection of unmarked or future archives remains unspecified.
- `ArrayStorage` stays generic: feature meanings, physical validity, identity
  ordering, and schema compatibility belong to producers/consumers.

Preserve the public paths `sawnergy.rin.RINBuilder`, `sawnergy.walks.Walker`,
`sawnergy.embedding.Embedder`, `sawnergy.embedding.align_frames`,
`sawnergy.visual.Visualizer`, and `sawnergy.embedding.Visualizer`. Reused utilities
retain their current module paths; V2 imports suitable helpers rather than
duplicating them. This does not extend the extraction milestone to new learning
components.

### Archive schema markers

**Writer postcondition:** Every archive successfully returned by the following
writers contains these three JSON-compatible root attributes after compression
and reopening:

| Writer | `schema_name` (string) | `schema_version` (integer) |
| --- | --- | --- |
| `RINBuilder.build_rin` | `"rin"` | `1` |
| `Walker.sample_walks` | `"rin_walks"` | `1` |
| `Embedder.embed_all` | `"shallow_residue_embeddings"` | `1` |

`producer_version` is the string `sawnergy.__version__` of the producing package.
It records which SAWNERGY release wrote the output; downstream writers stamp
their own output format and producer, rather than inheriting source markers.
The three schema names identify the corresponding archive contracts below.

Each schema is versioned independently, starting at integer `1`; the version
identifies the documented format, not the package release or Zarr format version.
A package-version change alone does not change a schema version. Channel choices,
window/batch sizes, and walk/embedding options within these contracts do not
create new schemas or schema versions.

This change adds only these attributes: existing datasets, values, metadata,
return values, and public signatures are retained. `ArrayStorage` does not
generate semantic markers automatically. Historical archives are not rewritten;
the policy for reading unmarked/unsupported versions and its validation are
separate migration items. No reader compatibility guarantee is introduced here.

### RIN archives (`RINBuilder.build_rin`)

Let `N` be the selected molecule's residue count and `T` the number of processed
frame batches. Public trajectory frame ranges are **1-based and inclusive**,
with `1 <= start <= end <= total_frames`. For resolved batch size `B > 0`,
`T = ceil((end - start + 1) / B)`; the last batch may be shorter. A nonpositive
`frame_batch_size` is resolved to the total trajectory frame count.
A stored timestamp therefore need not represent one original frame.

| Dataset-name attribute | Default block | Shape | Dtype / meaning |
| --- | --- | --- | --- |
| `com_name` | `COM` | `(T, N, 3)` | `float32`; batch-averaged residue centers of mass |
| `attractive_energies_name` | `ATTRACTIVE_energies` | `(T, N, N)` | `float32`; processed attractive magnitudes before normalization |
| `repulsive_energies_name` | `REPULSIVE_energies` | `(T, N, N)` | `float32`; processed repulsive magnitudes before normalization |
| `attractive_transitions_name` | `ATTRACTIVE_transitions` | `(T, N, N)` | `float32`; row-normalized attractive magnitudes |
| `repulsive_transitions_name` | `REPULSIVE_transitions` | `(T, N, N)` | `float32`; row-normalized repulsive magnitudes |

Energy blocks depend on `keep_prenormalized_energies`; either channel may also
be disabled. All five name attributes are written; unavailable blocks have
value `None`, while `com_name` is always `"COM"`.

The stored quantities depend on this processing order:

1. Add CPPTRAJ's batch-averaged atomic electrostatic and van der Waals maps.
2. Aggregate to residues as `E = P.T @ A @ P`, where `P[a, r]` indicates atom
   membership. Thus `E[r, s]` is the sum of `A[a, b]` over atoms in residues
   `r` and `s`.
3. Split into nonnegative magnitudes `max(-E, 0)` and `max(E, 0)`; prune each
   channel row below its `prune_low_energies_frac` quantile, zero the diagonal,
   and symmetrize by `(M + M.T) / 2`.
4. Optionally save the processed magnitudes. Normalize each channel row by
   `max(row_sum, 1e-12)` when its sum is positive; leave zero rows zero.

For finite inputs, prenormalized channels are symmetric, nonnegative, and have
zero diagonals. Transitions retain nonnegativity and zero diagonals but need not
be symmetric. Rows with prenormalized mass at least `1e-12` sum to one within
floating-point tolerance; smaller positive masses yield sums below one because
of the denominator floor. Unconditional row-stochasticity is not guaranteed.

Other root metadata: `time_created`, `molecule_of_interest`, `frame_range`,
`frame_batch_size`, and `prune_low_energies_frac`. `frame_range` retains the
supplied value (a JSON list or `None`), rather than always recording a resolved
range. The membership matrix sorts residue IDs and atom IDs numerically; COM
ordering comes from CPPTRAJ's residue loop. No explicit atom/residue identity
arrays, units, or original frame-ID array are stored. Public downstream node
IDs are **1-based local row positions**, not a persisted topology-ID mapping.

These combined, processed, batch-averaged channels do not substitute for raw
per-frame V2 features. V2 must separately specify physical meanings, units,
identity/order, frame correspondence, missing values, and its archive layout.

### Walk archives (`Walker.sample_walks`)

`T` and `N` come from the RIN transition stack. `L` counts steps, so a path has
`L + 1` node IDs. For `W = walks_per_node`, the per-node counts are
`S = int(round(W * saw_frac))` and `R = int(W) - S`.

| Dataset-name attribute | Default block | Shape |
| --- | --- | --- |
| `attractive_RWs_name` | `ATTRACTIVE_RWs` | `(T, N * R, L + 1)` |
| `attractive_SAWs_name` | `ATTRACTIVE_SAWs` | `(T, N * S, L + 1)` |
| `repulsive_RWs_name` | `REPULSIVE_RWs` | `(T, N * R, L + 1)` |
| `repulsive_SAWs_name` | `REPULSIVE_SAWs` | `(T, N * S, L + 1)` |

Blocks use `uint16` and 1-based local node IDs. Lossless storage consequently
requires `N <= 65535`; the writer does not explicitly enforce this bound.
Disabled channels have name attributes `None`. Enabled channels create both RW
and SAW blocks, including `(T, 0, L + 1)` when a count is zero. Within each
timestamp, paths are ordered by starting node, then repetition.

`walks_layout` is `"time_leading_3d"`, including time-aware walks: axis 0 then
denotes the starting timestamp, and intermediate timestamps are not stored.
SAW sampling can relax avoidance when remaining candidates have zero mass, so
the archive does not guarantee distinct nodes throughout every SAW.

Structural metadata: the four dataset-name attributes, `num_RWs`, `num_SAWs`,
`node_count`, `walk_length`, `walks_per_node`, and `time_stamp_count`.
Other metadata: `time_created`, `seed`, `rng_scheme`
(`"SeedSequence.spawn_per_batch_v1"`), `num_workers`, `in_parallel`, and
`batch_size_nodes`. Reproducibility depends on batching configuration as well as
seed; serial/parallel equality is not promised. Time-awareness, stickiness, and
original trajectory identities are not recorded by this writer.

### Shallow embedding archives (`Embedder.embed_all`)

This is an existing V1 format, not a V2 learning specification.
`frame_embeddings_name = "FRAME_EMBEDDINGS"` resolves a `float32` block shaped
`(T, N, D)`, preserving walk timestamp/local node order. `embedding_kind` is
`in`, `out`, or `avg`; `objective` is `sg` or `sgns`. Each model after the first
is warm-started from the preceding frame's input/output weights; `embed_all`
does not automatically call `align_frames`.

Root metadata written by this producer:

- Layout: `frame_embeddings_name`, `time_stamp_count`, `node_count`, `embedding_dim`.
- Provenance: `source_WALKS_path`, `walk_length`, `num_RWs`, `num_SAWs`, and the
  four source walk dataset-name attributes.
- Training: `objective`, `model_base`, `embedding_kind`, `num_epochs`,
  `batch_size`, `window_size`, `alpha`, `negative_sampling`,
  `num_negative_samples`, `lr_step_per_batch`, `shuffle_data`, `device_hint`,
  `model_kwargs_repr`, `RIN_type`, and `using`.
- Reproducibility/storage: `master_seed`, `per_frame_seeds`, `arrays_per_chunk`,
  `compression_level`, and `created_at`.

`device_hint` is empty when unspecified; `model_kwargs_repr` is provenance text,
not an executable serialized configuration. Archive/seed/warm-start tests do not
establish biological meaning or cross-frame alignment.

## Reusable CPPTRAJ components (`rin_util.py`)

### `CpptrajScript`

**Invariant:** an ordered tuple of command strings. The frozen dataclass prevents
field reassignment; composition returns new objects without mutating operands.
Constructor annotations do not enforce tuple/string contents.

| Operation | Preconditions | Postconditions / failures |
| --- | --- | --- |
| `from_cmd(cmd)` | String command | New script with `commands == (cmd,)` |
| `script + other` | String or `CpptrajScript` | Append one command or concatenate command tuples in order; unsupported operands return `NotImplemented` from `__add__`, normally resulting in `TypeError` |
| `script > target` | For a filename, nonempty script and CPPTRAJ-compatible target string | Append `" out <target>"` to the last command in a new script; a script target instead means concatenation. Empty script plus filename raises `IndexError` |
| `render()` | Valid command lines / CPPTRAJ syntax | Join with newlines and a trailing newline; append `run` only if the exact string `"run"` is absent from `commands` |

An empty script renders as `"run\n"`. Rendering does not add `quit`, parse
commands, normalize case/whitespace for `run`, or quote paths/masks. Callers
supply complete lines and valid CPPTRAJ arguments.

`PAIRWISE_STDOUT` supplies `run` followed by
`printdata PW[EMAP] square2d noheader` and
`printdata PW[VMAP] square2d noheader`. `COM_STDOUT(mol_id)` supplies `run`, the
residue dataset-coordinate loop, and `printdata COMX* COMY* COMZ*`. These require
matching upstream dataset names and CPPTRAJ syntax. They neither parse nor
validate output and are not generic V2 feature extractors.

### `locate_cpptraj(explicit=None, verify=True) -> str`

**Preconditions:** `explicit` is `None` or a `Path` to an executable candidate;
the implementation calls `.resolve()` on it before candidate construction.
An environment directory is not an executable path: supply its `bin/cpptraj`.

**Postconditions:** Return the resolved absolute path of the first existing,
executable candidate that, with verification enabled, returns code 0 or 1 for
`cpptraj -h`. Search order: explicit path, `CPPTRAJ`, PATH searches for
`cpptraj` and `cpptraj.exe`, `$AMBERHOME/bin/cpptraj`, `$CONDA_PREFIX/bin/cpptraj`.
Equal `Path` candidates are deduplicated in first-seen order before resolution;
aliases to the same executable need not be deduplicated. An invalid explicit
candidate permits fallback.

**Failures/limits:** Verification has a 15-second timeout per candidate;
timeouts skip candidates, despite the docstring listing `TimeoutExpired` as a
raised error. Exhaustion raises `CpptrajNotFound` (a `RuntimeError` subclass)
listing candidates. Other launch errors propagate. A help probe does not verify
feature support or numerical-output compatibility.

### `run_cpptraj(cpptraj, script=None, argv=None, timeout=None, *, env=None)`

**Preconditions:** Runnable executable; argument strings in `argv` if supplied;
text `script` if supplied; `timeout=None` or a positive subprocess time limit.
Script text contains complete CPPTRAJ commands on separate lines; a final
newline is optional.
`env=None` inherits the environment; a supplied dictionary is a complete child
environment with string keys/values. Use `create_updated_subprocess_env` for
inherited overrides. Script and argv may both be supplied.

**Postconditions:** Run `[cpptraj] + (argv or [])` without a shell, capture
stdout/stderr as text, and return stdout unchanged on exit code zero. Send the
script on stdin. For termination detection, strip `#` comments from each line,
ignore blank/comment-only lines, and compare the last remaining line with the
standalone command `quit`, ignoring surrounding whitespace and case. If it is
already terminated, forward the original script unchanged, including comments
and whitespace. Otherwise append `quit\n` on its own line, inserting a separating
newline when the nonempty script does not end in `\n`. An empty script becomes
`"quit\n"`; with `script=None`, stdin input remains `None`. A filename, dataset
name, command argument, or comment ending in `quit` is not a terminator. The
caller's argument list, environment, and script are not mutated.

**Failures/limits:** Nonzero exit raises `subprocess.CalledProcessError` with
captured output; timeout and launch errors propagate. Successful exit does not
validate datasets or physical values. Termination detection is a line-based
check, not a general parser for multiline/continued commands or quoted script
blocks. Callers remain responsible for valid CPPTRAJ syntax. Tests check exact
stdin, argument/environment forwarding, and preservation of subprocess failures.

### `CpptrajMaskParser.hierarchize_molecular_composition(mol_compositions_file)`

**Preconditions:** Readable UTF-8 table with bracketed `[Mnum]`, `[Rnum]`, and
`[AtNum]` labels in its first line, and integer fields at those positions in
every nonblank data row. Labels should be unique; duplicate labels are not
rejected (the last occurrence wins).

**Postconditions:** Return `{molecule_id: {residue_id: set(atom_ids)}}` using
unchanged integer IDs. Ignore blank data lines; repeated rows collapse through
set insertion. Dictionary insertion order follows first appearance; atom sets
provide no ordering. A header-only file returns `{}`. Parsing does not sort,
renumber, validate positive IDs, enforce unique atom ownership, or check
correspondence with other arrays. V2 must establish these properties explicitly
where it constructs identity/order mappings.

**Failures:** Empty file raises `RuntimeError`; missing required columns,
insufficient row fields, and noninteger IDs raise `ValueError`. File errors
propagate. The legacy `com_parser` is not needed for this reuse boundary: it
requires exactly seven whitespace-separated fields and emits the first four
as CSV without numerical validation. Its docstring's weaker field-count claim
does not describe its implementation.

## Reusable storage components (`sawnergy_util.py`)

### `ArrayStorage(pth, mode)` and lifetime

**Preconditions:** String or `Path` and one of `r`, `r+`, `a`, `w`, or `w-`.
Unsupported modes raise `ValueError` before acquiring a store. Exact lowercase
suffix `.zip` selects an existing ZIP and requires `mode="r"`. Otherwise the
suffix is replaced with `.zarr`; stored paths are resolved.

| Local-store mode | Opening behavior |
| --- | --- |
| `r` | Open an existing group read-only; do not create directories, initialize attributes, or otherwise write metadata |
| `r+` | Open an existing group for updates; do not create a missing store/group |
| `a` | Open an existing group or create a missing group |
| `w` | Create a group, replacing existing contents |
| `w-` | Create a group; fail if the store already contains data |

Writable opens initialize missing storage metadata mappings without replacing
existing mappings or unrelated attributes. Read-only opens accept groups missing
these mappings and leave their attributes untouched; missing attributes still
raise `KeyError` through `get_attr`. `ArrayStorage` does not add semantic schema
validation or decide archive compatibility here.

**Postconditions/invariants:** Expose `root`, `store`, `store_path`, and `mode`;
root is a group. Blocks created through `write` retain fixed item shape, dtype,
and axis-0 chunk length recorded in the three storage mappings. Direct root
mutation bypasses helper invariants. Storage does not enforce finite numbers,
physical units, identity maps, or agreement of semantic metadata with arrays.

**Checks/limits:** Invalid path types/non-read ZIP modes raise `ValueError`;
missing ZIP files and missing local stores in `r`/`r+` raise `FileNotFoundError`.
A root array encountered in `r`, `r+`, or `a` raises `TypeError`; it is not
converted to a group. Opening errors and metadata-initialization errors propagate
without retrying in another mode. If initialization fails after acquiring a
store, attempt `close()` before propagating the original exception, including
interruption. Closing is best-effort: ordinary closing errors are logged and do
not replace the initialization error. Zarr opening modes are not transactional;
for example, `w` may already have cleared old data before a subsequent failure.
`close()` suppresses store-closing errors. Context exit closes the store without
suppressing the body exception. No operation-after-close guarantee is specified.

**Local dependency limitation:** With Zarr 3.1.2, reopening a valid Zarr v3 group
under `python -O` fails inside Zarr: `GroupMetadata.from_dict` removes `node_type`
inside an `assert`, whose side effect is skipped under optimization. SAWNERGY's
explicit constructor checks remain active, but this dependency prevents complete
optimized-mode archive workflows in that environment. Do not bypass or suppress
the resulting metadata error; dependency compatibility requires separate work.

### Blocks and attributes

| Public operation | Preconditions | Successful postconditions / explicit failures |
| --- | --- | --- |
| `write(these_arrays, to_block_named, *, arrays_per_chunk=None)` | Writable store; finite homogeneous batch or stacked ndarray `(K, *item_shape)`; stable shape/dtype across appends; positive integer chunk length if supplied | Append `K` items in order, retaining old rows; return `None`. Empty input creates no block. Read mode raises `RuntimeError`; within-batch shape/dtype differences raise `ValueError`; cached shape/dtype/chunk conflicts raise `RuntimeError` |
| `read(from_block_named, ids=None)` | Existing array with an axis 0; valid axis-0 integer, slice, or 1D integer index collection | Detached NumPy copy with stored dtype. `None` reads all; an integer drops axis 0; slices/collections retain it. Collections use orthogonal indexing after NumPy index coercion. Missing name raises `KeyError`; non-array member raises `TypeError`; selection errors propagate |
| `block_iter(from_block_named, *, step=1)` | Existing array; positive integer step | Detached copies of consecutive slices, at most `step` rows each, covering axis 0 in order, with dtype and trailing dimensions preserved. Zero/negative steps raise `ValueError("step must be positive")`, including for empty/scalar arrays. Missing/non-array members raise `KeyError`/`TypeError`; scalar arrays are yielded once |
| `list_blocks()` | Open store | Sorted names of immediate root arrays; no recursive subgroup listing |
| `delete_block(named)` | Writable store; existing block | Remove member and entries from all three storage mappings; read mode raises `RuntimeError`, missing name `KeyError` |
| `add_attr(key, val)` | Writable store; string key; JSON-compatible value after conversions below | Set/replace root attribute; read mode raises `RuntimeError`; unsupported values may raise serialization errors |
| `get_attr(key)` | Existing attribute | Return stored JSON-compatible value; missing key raises `KeyError` |

Attribute conversion is top-level: NumPy scalars via `.item()`, arrays via
`.tolist()`, sets/tuples to lists, dates/datetimes to ISO strings. Nested
non-JSON objects are not recursively converted. Original tuple/set types are
not restored; set order is not guaranteed. Callers must not overwrite storage
mappings with inconsistent metadata.

First-write omission of `arrays_per_chunk` defaults to 10 with `RuntimeWarning`;
later omission reuses the stored value. Empty writes in read mode still raise.
For nonempty writes, validate every item's shape/dtype and materialize the batch
before creating a block or changing metadata. The materialized shape must be
`(len(these_arrays), *item_shape)`; a custom conversion cannot silently add/drop
items or change their shape. Resolve cached shape/dtype/chunk
settings without mutation, and check an existing member's array type, axis-0
presence, item shape, dtype, and axis-0 chunk length before filling missing
metadata. Cached conflicts retain `RuntimeError`; incompatible existing member
type/shape/dtype raises `TypeError`, and an incompatible existing chunk length
raises `RuntimeError`. Resolved chunk lengths must be positive.

**Failure postcondition:** Rejected shape, dtype, or chunk arguments, batch
conversion errors, and warnings promoted to errors leave array contents, shapes,
members, and metadata unchanged. In particular, validation cannot leave a new
empty block or partially filled metadata mappings. Empty input remains a no-op
without chunk validation or block creation.

After validation succeeds, create any new block and record its metadata, then
resize and assign data. These storage mutations are **not transactional**: an
I/O/codec failure during creation, metadata persistence, resize, or assignment
may leave partial state. This patch does not add rollback for such failures.

`block_iter` validates when iteration begins, as with other generator errors.
For positive steps, empty arrays yield no chunks; scalar arrays yield one scalar
NumPy array. Modifying a yielded chunk does not modify storage. Writes and appends
preserve zero-length item dimensions in array shapes and metadata while keeping
chunk dimensions positive, as described in the common archive layout above.

### `compress(into=None, *, compression_level)`

**Preconditions:** Open store; for a local source, writable destination and a
level convertible by `int` to `[0, 9]`. For predictable reuse supply an integer
level and explicit `.zip` destination. Supported layout is immediate arrays,
not nested groups. Each source array is materialized in memory when copied.

**Postconditions:** Copy root attributes and immediate arrays with their values,
shapes, dtypes, chunks, and attributes to a ZIP; return its path string. Use
Blosc Zstd/shuffle at the requested level (0 still uses a Blosc container).
Source remains available. `into=None` uses the source path with suffix `.zip`;
explicit `.zip` suffix matching is case-insensitive; otherwise `into` is a
directory. Explicit destination parents are created as needed.

For a ZIP source, return its existing path without copying or validating level;
`into` is ignored. For a local source, build a temporary ZIP file in the
destination's directory and close it successfully before atomically replacing
the destination. Successful calls still replace existing output. Readers see
either the previous complete file or the replacement, rather than a partially
written destination.

Invalid levels raise conversion errors or `ValueError`; creation, copying,
closing, and replacement errors propagate. Failures before successful replacement
leave an existing destination byte-for-byte unchanged, or leave no destination
if none existed. Attempt to remove temporary output on failure, including
interruption; ordinary removal errors are logged without replacing the original
exception. Cleanup cannot be guaranteed against filesystem errors. Atomic
replacement does not promise durability across power loss or preserve ZIP bytes,
original codec settings, or destination filesystem metadata.

### `ArrayStorage.compress_and_cleanup(output_pth, compression_level)`

**Preconditions:** Same destination/level obligations as compression.
**Normal postconditions:** Yield one writable temporary storage; compress only
after the body completes successfully, then close it and remove its temporary
directory. The public signature and successful archive layout are unchanged.

**Failure postconditions:** A body exception (including interruption) skips
compression, leaving the destination untouched. Attempt store closure on both
body and compression failure, and temporary-directory cleanup on every exit,
including storage-construction failure. Body/compression exceptions propagate;
cleanup errors are logged rather than replacing an active error. With no active
error, a temporary-directory cleanup failure propagates. `close()` retains its
existing best-effort behavior (ordinary store-closing errors are logged).

Archive publication uses the safe replacement behavior of `compress` above.
Cleanup is attempted, not guaranteed against filesystem failures.

## Other reusable helpers (`sawnergy_util.py`)

### `elementwise_processor(in_parallel=False, Executor=None, max_workers=None, capture_output=True)`

**Preconditions:** Runner receives a finite iterable and callable with matching
extra arguments. Parallel execution requires an executor class and valid worker
count (use `None` or a positive integer). Process execution also requires
picklable callables/arguments and a main-entry guard. Callers manage shared
mutable state.

**Postconditions:** On success, apply `function(item, *args, **kwargs)` once to
each item. Capture mode returns a list in **input order**, including `[]` for
empty input; otherwise wait for execution and return `None`. Parallel side
effects need not follow input order. Missing executor raises `ValueError` upon
runner invocation, not factory creation. `None` or a falsy worker value resolves
to `os.cpu_count() or 1`.

Exceptions propagate in both capture modes. Already-started side effects are
not rolled back; submitted work need not stop immediately. Parallel mode submits
all input items before collecting results, so this is not bounded streaming.
`is_main_process()` means no parent process and process name `"MainProcess"`;
it does not itself establish that application code has a main-entry guard.

### `batches_of(iterable, batch_size=-1, *, out_as=list, ranges=False, inclusive_end=False)`

**Preconditions:** Finite iterable, integer batch size, and constructor `out_as`
accepting a batch or index pair.

**Postconditions:** Positive size yields consecutive nonempty batches of at most
that size, preserving each item once and in order. Nonpositive size yields one
batch for nonempty input. Range mode returns **0-based consumption positions**,
not input values: `(start, stop_exclusive)` or `(start, end_inclusive)`, wrapped
with `out_as`. Thus `batches_of([10, 20, 30], 2, ranges=True)` yields `[0, 2]`,
`[2, 3]` because the default wrapper is `list`.
`inclusive_end` does not affect data batches.

**Empty-input postcondition:** Empty sequences, NumPy arrays, iterators, and
generators yield nothing for every valid batch size, including the default,
zero, and negative sizes. This applies to data batches and exclusive/inclusive
ranges. There is no empty batch, fabricated range, zero-step error, or call to
`out_as` for empty input.

Sliceable inputs are sliced; generic inputs are consumed into lists, including
in range mode. Nonpositive sizes provide no bounded-memory guarantee. These index
ranges are not CPPTRAJ's 1-based inclusive frame ranges. The signature, default
wrapper, and nonempty batching/range conventions are preserved.

### Small file/environment/pipeline helpers

| Helper | Preconditions | Postconditions / limits |
| --- | --- | --- |
| `read_lines(file_path, skip_header=True)` | Readable UTF-8 file | Lines retaining newline characters (standard text-mode universal newline conversion applies), optionally removing exactly the first; empty file gives `[]`; file/decoding errors propagate |
| `temporary_file(prefix, suffix)` | Valid filename components; writable system temporary directory | `Path` to a new empty file with its handle already closed; caller owns deletion, including on failure |
| `create_updated_subprocess_env(**var_vals)` | String environment names; values convertible to strings (path-like values must produce strings for the annotated result) | Copy `os.environ`; remove keys with `None`; booleans become `"TRUE"`/`"FALSE"`, path-like values use `os.fspath`, others `str`; parent environment and omitted variables remain unchanged |
| `compose_steps(*steps)` | Each step is `(callable, kwargs_or_None)`; kwargs convertible by `dict`; compatible result/input types | Unary pipeline in supplied order; no steps returns input object unchanged; shallow-copy kwargs per call; propagate conversion/step errors; input mutation is allowed and a returned `None` becomes the next input |
| `current_time()` | None | Local wall-clock text `%Y-%m-%d_%H%M%S`; no timezone, uniqueness, or monotonicity guarantee |

`compose_steps` checks tuple length/callability at construction (`TypeError`),
but kwargs only at invocation. Copies are shallow: nested mutable values remain
shared. Legacy memory-based file/directory chunking, private V1 matrix/walk
internals, and neural backend implementations are outside this reuse scope.

## V2 feature interfaces and representation contracts (design draft)

This section precedes implementation and tests. Signatures and names below are
proposed V2 interfaces, not changes to existing V1 public APIs. The agreed design
uses batch feature callables, incremental archive writing, and explicit feature
dependency registration. Output samples correspond to explicit source-frame
windows (tiles); a single-frame sample is a window containing one frame.
Computational batching is separate from the declared window statistic.

Keep `FeatureExtractor` and its execution context in `feature_extractor.py`;
keep feature specifications, frame representations, and frame/trajectory
validation in `reprs.py`. No base-class hierarchy or feature subclasses are
required: a feature is a callable plus its specification.

### Shared extraction context and selection

`selection` maps CPPTRAJ molecule IDs to lists of selected topology residue IDs,
for example `{1: [10, 12, 15], 3: [40, 45]}`. Proposed input rules:

- Selection is nonempty; each molecule has a nonempty list of distinct residues.
  IDs are positive integers, excluding booleans. Unknown IDs or residues assigned
  to the wrong molecule raise `ValueError`; invalid input types raise `TypeError`.
- Preserve mapping iteration order, then each residue list's order. Flatten to
  an immutable tuple of `(molecule_id, residue_id)` pairs. For the example, array
  positions correspond to `((1, 10), (1, 12), (1, 15), (3, 40), (3, 45))`.
- Resolve all atoms of each selected residue from the topology. Keep one sorted
  tuple of atom IDs per residue in that same residue order. Validate membership;
  do not silently discard unavailable residues or select only some of their atoms.
- Include cross-molecule residue pairs in pairwise feature arrays. Preserve the
  original topology IDs separately from zero-based NumPy array positions.

The extractor establishes this mapping once, independently of feature outputs,
and uses it to construct its contracts. Keep the existing
`CpptrajMaskParser.hierarchize_molecular_composition` signature and behavior;
selection handling belongs in a new helper or method.

`ExtractionContext` is a small immutable record supplied to feature callables:

| Field | Meaning |
| --- | --- |
| `cpptraj` | Resolved executable path |
| `topology_file`, `trajectory_file` | Source paths |
| `windows` | Ordered source-frame windows for the current batch; each window is a tuple of original 1-based frame IDs |
| `residue_ids` | Ordered `(molecule_id, residue_id)` tuples described above |
| `atom_ids` | Atom-ID tuples aligned with `residue_ids` |
| `subprocess_env`, `timeout` | Child environment and time limit forwarded to CPPTRAJ helpers |

Each window is nonempty, contains distinct frame IDs, and lies within the source
trajectory's frame range. A processing batch contains a consecutive slice of the
complete requested window sequence; no window is silently omitted, duplicated,
or changed. All features use the same source frames for each output sample.
Residue/atom identity mappings and source settings stay fixed for an extraction.
Context collections are copied/frozen on construction; callables do not mutate them.

### Window statistics and CPPTRAJ output semantics

Every feature-specific implementation returns one result per requested window,
implementing a declared mathematical statistic. It handles the backend as follows:

1. If the selected CPPTRAJ output already provides the required window statistic,
   use it directly.
2. If it provides per-frame values, accumulate and reduce those values in Python
   to obtain the declared statistic.
3. If neither route can produce the declared statistic, reject that configuration
   with `ValueError`; do not silently substitute a different quantity.

Specifying a [`trajin`](https://amberhub.chpc.utah.edu/trajin/) range selects
input frames; it does not universally request
averaging. Semantics belong to the particular action/output. For example, the
documented [`pairwise`](https://amberhub.chpc.utah.edu/pairwise/) EMAP/VMAP
matrices contain averages over processed frames, whereas
[`vector ... center`](https://amberhub.chpc.utah.edu/vector/) supplies a
trajectory of COM coordinates. Native
averaging must use exactly the requested window and denominator. Processing
several windows in one CPPTRAJ run does not automatically yield one summary per
window. Verify backend behavior against the installed CPPTRAJ with reference tests.

The base extractor validates and stores window results uniformly; it does not
guess whether an array has already been reduced, or apply an extra temporal
reduction to native summaries. Raw intermediate values need exist only while
processing their window; retaining all original frames in the archive is not
required. Processing batch size must not change the declared physical statistic.

If a mean is computed from partial batches with counts `n_b` and means `mu_b`,
combine as `sum(n_b * mu_b) / sum(n_b)`, using actual processed frame counts.
An unweighted mean of batch means is generally incorrect for unequal counts.
Other statistics require their own valid combination rules. Record each window's
source frames, count, feature statistic, and any weighting/normalization policy.
Missing frames are errors, not a reason to silently change the denominator.

Aggregation order is part of the feature definition. Mean per-frame COM distance
is not generally the distance between mean COM coordinates. Similarly, mean
hydrogen-bond count is distinct from the fraction of frames with any bond.
Dependencies exposing only window summaries cannot replace per-frame information
needed for a different statistic. An adapter may compute raw intermediates locally;
cross-feature reuse of those intermediates needs an explicit interface before
implementation, rather than being inferred from summary arrays.

### Feature definition and registration

`FeatureSpec` is the declaration used to generate validation rules:

```python
FeatureSpec(
    shape,       # Per-output-sample tuple: integer dimensions and the symbol "N".
    dtype,       # Explicit NumPy real numeric/bool dtype; no implicit default.
    units,       # Nonempty unit string; "1" for dimensionless quantities.
    statistic,   # Explicit window quantity, including aggregation order.
    requires=(), # Names of explicitly registered dependencies.
    validate=None,
)
```

`"N"` resolves to the selected residue count. Shape dimensions are nonnegative
integers or `"N"`; residue features include at least one residue axis. The first
features use `("N", 3)` for COMs and `("N", "N")` for pairwise distances and
energy channels. All residue axes refer to the same ordered selection. A feature
has a fixed resolved shape, dtype, units, and statistic throughout one trajectory.
Specifications and generated contracts snapshot their declarations and identity
collections; later mutation of caller-owned input mappings must not change them.

The optional feature-specific validator has this proposed signature:

```python
validate(values: np.ndarray, dependencies: Mapping[str, np.ndarray]) -> None
```

It checks one output sample at a time, after generic shape/dtype/finiteness checks.
`dependencies` contains that window's arrays for exactly the declared dependency
names. Successful validation returns `None`; a failed condition raises an error.
Validators must not mutate inputs. This permits physical and cross-feature rules
without putting feature names or physics into the base extractor.

Proposed extractor registration interface:

```python
add_feature(self, name: str, extractor: FeatureCallable, specification: FeatureSpec) -> None
supported_features(self) -> tuple[str, ...]
```

- Registry state belongs to each instance. Registration never adds class
  attributes or changes other instances. `supported_features` reports this
  instance's registered names in registration order.
- Names are nonempty, unique strings. The extractor is callable. Specifications
  must be valid before registration modifies the registry; failed registration
  leaves it unchanged.
- Dependencies are distinct existing feature names. Register dependencies first;
  missing/self dependencies and duplicate feature names raise `ValueError`.
  Nothing is registered or computed implicitly. Registration order consequently
  supplies an execution order without a separate dependency scheduler.
- Registration is fixed during extraction. At least one feature must be
  registered. All registered features, including dependencies, are output
  channels in this initial design; there is no separate hidden-feature mechanism.
- The incomplete V2 class-level `new_feature` stub is superseded by this proposed
  instance-level interface; V1 public APIs remain unchanged.

### Feature callable

`FeatureCallable` denotes a callable with the following signature:

```python
extract_feature(
    context: ExtractionContext,
    dependencies: Mapping[str, np.ndarray],
) -> np.ndarray
```

**Preconditions:** The context has valid, independently resolved selection and
window metadata. `dependencies` contains exactly the registered feature's
`requires` names. Each dependency has already passed validation for this same
batch, window order, and residue order, with its declared statistic available to
the adapter. Dependency arrays are borrowed read-only
inputs; they must not be mutated or retained for later mutation.

**Postconditions:** For `B = len(context.windows)`, return a NumPy array with
shape `(B, *resolved_sample_shape)`, the exact declared dtype, and finite
values. Slice `result[k]` is the declared statistic for `context.windows[k]`.
Every residue axis follows `context.residue_ids`. The callable relinquishes the
returned data to the extractor and must not subsequently modify it. There is no
silent dtype conversion, padding, missing-value substitution, or undeclared aggregation.

The extractor checks the generic contract and applies the feature-specific
validator to every output sample before writing its values. Wrong return types/dtypes
raise `TypeError`; invalid shapes/non-finite values raise `ValueError`.
CPPTRAJ, parser, and custom callable/validator failures propagate and abort
extraction. A callable owns cleanup of temporary resources it creates.

Backend adapters must validate and map their source frame/entity identifiers.
If CPPTRAJ reports run-local indices, their relationship to original frame IDs
must be explicitly established and tested. Assigning expected labels based only
on row counts is insufficient. Generic contracts cannot prove that numbers
actually refer to their declared atoms/residues; backend reference tests are
required for that semantic guarantee.

### Frame representation and contract

```python
FrameRepresentation(source_frame_ids, residue_ids, features)
FrameContract(residue_ids, features)  # features: Mapping[str, FeatureSpec]
FrameContract.validate(self, frame: FrameRepresentation) -> None
```

Here a frame representation means one output sample, potentially summarizing a
window. It contains the original source-frame ID tuple, the ordered residue
identity tuple, and a mapping from registered names to sample arrays. A singleton
source tuple identifies an original frame; an aggregate is not labeled as though
it were one original frame. Units and statistic declarations live in the contract
rather than being repeated per sample.
The frame contract is generated from the independently resolved selection and
registered specifications, never inferred from arrays being checked.

**Validation postconditions:** Successful validation establishes that source-frame
IDs are nonempty, distinct positive integers, residue identities equal the
expected tuple, feature names match exactly, and every feature satisfies its resolved shape, declared
dtype, finiteness, and feature-specific validator. A frame contract alone does
not establish that this sample occupies the correct trajectory position or that
its source window matches the independently requested window.

Representations can borrow slices of the current batch for validation and
writing; they need not copy or accumulate all samples. Their validity is not a
permanent guarantee if a caller later mutates an array. The extractor does not
modify a frame between validation and storage.

### Trajectory representation and contract

```python
TrajectoryContract(windows, frame_contract)
TrajectoryContract.validate(self, storage: ArrayStorage, *, batch_size: int = 1) -> None
```

The trajectory representation is the named arrays and metadata in `ArrayStorage`;
no additional class holding every frame in RAM is required. The trajectory
contract holds the complete requested window sequence and one shared frame
contract. The window sequence is nonempty, and every window is validated against
the source frame count before extraction. Window construction, stride, overlap,
and final partial-window policies must be explicit before implementing selection.

**Trajectory validation establishes:**

- Stored source windows and their counts match the requested sequence exactly,
  including order; no missing, duplicated, unexpected, or relabeled samples.
- Stored residue identities match the contract's ordered identity tuple, and
  feature names, shapes, dtypes, units, statistics, and dependency declarations
  agree with their expected specifications.
- Every feature block has shape `(S, *resolved_sample_shape)`, where
  `S = len(windows)`. There are no shorter/longer channels.
- Every stored sample passes the shared frame contract. Validation reads values
  in bounded batches (`batch_size` must be a positive integer), including
  cross-feature dependency checks; it does not load the full trajectory at once.

Validation returns `None` on success and raises on a mismatch. Metadata and
structural mismatches are checked before loading feature values. Physical
validators are registered application code; archive text is never executed or
imported to reconstruct them. Exact dataset/metadata keys and schema markers
must be specified with the archive format before implementing this validator.

### Extraction lifecycle and testing boundary

1. Validate registration, source-window selection, and residue membership; build
   the frame and trajectory contracts from these independent inputs.
2. Enter `ArrayStorage.compress_and_cleanup` and process requested window batches
   in order. Execute explicitly registered features in registration order.
3. Validate every sample against its expected source window and frame contract,
   then append its feature data to temporary storage. Store actual identity/unit
   metadata consistently with the contracts.
4. Validate the completed temporary trajectory against its contract, reading
   stored values incrementally. This adds a verification read before compression.
5. Exit normally to compress and publish. A failed extraction or validation
   raises inside the context, so output is not published and cleanup is attempted.

Batch size controls extraction working data; windows and feature statistics
control temporal aggregation. No design guarantee is made about an individual
backend's atomic-matrix memory use. Feature
results/dependencies for completed batches can be released after writing.

Use `icontract` for pre/postconditions and object invariants when implementing
the stubs. Shared validation predicates should avoid divergent definitions.
Essential caller-input, external-output, and archive checks must execute under
`python -O`; decorators alone must not be their only enforcement. Frozen records
do not make NumPy array contents immutable or prove numerical correctness.

Before implementation, write tests for instance isolation, failed-registration
state, dependency ordering, window/batch alignment, frame/trajectory mismatches, and
failed validation preventing publication. Use small synthetic features for
interface tests; use independently computed reference examples for physical
extractors. Include equivalence tests for native and Python reductions where both
are supported, unequal partial-batch counts, unsupported statistics, and rejection
of summary dependencies that cannot supply a requested statistic. Exact built-in
units/dtypes, energy definitions and diagonal
conventions, numerical tolerances, and archive field names still require their
own specifications before those components are implemented. Raw COM distances are
Euclidean distances between per-frame COMs using the agreed V1 coordinate
processing procedure; their window statistic must specify aggregation order.
Electrostatic/vdW channels remain separate and signed.
The COM backend specification must also address imaging/unwrapping state at batch
boundaries, so a computational partition does not inadvertently change the
physical quantities being extracted. Hydrogen-bond definitions remain future work.

## Verification boundary

`tests/test_archive_schemas.py` checks the producer and schema markers after
ZIP reopening for all three writers, including single/both-channel RINs,
mixed/empty walk channels, time-aware walks, and the embedding pipeline. It also
checks that generic `ArrayStorage` creation does not add semantic markers.

Tests in `tests/test_storage.py` compare actual stored values for stacked writes
and appends, including zero-length item dimensions. They cover chunk defaults,
selection values/order/shape/dtype and detached copies, selection errors,
non-array members, immediate-root listing, successful deletion and metadata
removal, attribute conversion/replacement/persistence, and read-only mutators.
Write-validation tests compare in-memory metadata and stored files before/after
rejected calls, including partial metadata, incompatible members/chunks, batch
conversion, and warnings promoted to errors. Valid append/input forms and empty
no-ops are also checked.
Constructor tests cover read-only local/ZIP access without attribute writes,
missing stores/groups, all supported opening modes, root-array rejection, and
closure/error preservation on failed initialization.
They also cover context-manager cleanup and exception preservation, complete
archive publication, destination preservation on compression failures, temporary
output cleanup, and existing destination-path/ZIP-source behavior.
`tests/test_rin.py` checks immutable script composition/rendering, mask-table
IDs/order/duplicates and malformed input, executable candidate priority,
deduplication, timeout fallback, verification bypass, exhaustion, and launch errors.
It also checks exact script termination, subprocess argument/environment
forwarding, rejection of non-finite pairwise output, and unchanged exception
propagation. Synthetic RIN archive tests cover both channels, disabled blocks,
frame selection, COM means including a short final batch, metadata, sorted
atom/residue aggregation, and normalization's zero rows/denominator floor.
`tests/test_sawnergy_util.py` checks empty sequences, arrays, iterators, and
generators across batch sizes and range modes, plus preservation of nonempty
batch contents, range endpoints, and output wrappers. It also checks sequential
and thread-executor ordering/completion, extra arguments, empty input, worker
defaults, missing executors, and exception propagation; environment isolation and
conversion; file reading and temporary-file ownership; pipeline identity/order,
kwargs timing, shallow sharing, `None` results, and failures; and clock/process
helpers.
Walk tests check every node's transition vector in every frame, archive block
names/shapes/dtypes/counts, 1-based starts in repetition order, empty enabled
channels, disabled channels, and time-aware layout. Embedding tests check ordered
values, archive metadata, persisted frame seeds, seed forwarding, and warm starts.
Plotting fixtures replace GUI setup. CPPTRAJ calls and pipeline shared memory
are mocked; these checks do not establish native extraction or process-worker
correctness.

These tests are not complete coverage of the contracts above. Before changing
reused code, add contract-based tests for its relevant success, boundary, and
failure cases, including preservation of existing public behavior. Real CPPTRAJ
extraction checks remain separate from the existing mocked suite; storage
failure tests do not establish power-loss durability or guaranteed cleanup when
the filesystem refuses removal.

V2 interface/representation contracts remain design drafts; tests for them are
required before their implementations, not placeholder tests against absent APIs.
The local final pass skips the Torch backend tests because the installed Torch
cannot load `libtorch_cpu.dylib`; passing stub-based archive tests do not validate
that backend. The optimized-mode Zarr limitation above remains a dependency
issue outside this prerequisite queue.
