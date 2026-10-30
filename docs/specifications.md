# Specifications

## Scope and status

This document specifies the existing archive formats and reusable public behavior
needed for V2 feature extraction. It is based on `src/sawnergy/rin/rin_util.py`,
`src/sawnergy/sawnergy_util.py`, and the current V1 archive writers/readers. It
does not specify every V1 internal or define the new V2 feature format.

**Preconditions** are caller obligations, not claims that every obligation is
currently checked. **Postconditions** apply after successful return.
**Invariants** describe valid objects/data. Explicit checks and limitations are
identified separately. No `icontract` decorators are added by this document;
essential input/archive validation must remain active under `python -O` when
contracts are implemented. Known defects below are recorded for tests and later
fixes, not requirements to preserve them.

Schema markers, the policy for unmarked archives, storage failure guarantees,
and new V2 extraction contracts remain pending tasks in `V2_migration.md`.

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
- Current writers do not add `producer_version`, `schema_name`, or
  `schema_version`; readers do not validate these markers. This baseline does
  not decide acceptance/rejection of unmarked or future archives.
- `ArrayStorage` stays generic: feature meanings, physical validity, identity
  ordering, and schema compatibility belong to producers/consumers.

Preserve the public paths `sawnergy.rin.RINBuilder`, `sawnergy.walks.Walker`,
`sawnergy.embedding.Embedder`, `sawnergy.embedding.align_frames`,
`sawnergy.visual.Visualizer`, and `sawnergy.embedding.Visualizer`. Reused utilities
retain their current module paths; V2 imports suitable helpers rather than
duplicating them. This does not extend the extraction milestone to new learning
components.

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
A script must end in a newline unless it already terminates in `quit`.
`env=None` inherits the environment; a supplied dictionary is a complete child
environment with string keys/values. Use `create_updated_subprocess_env` for
inherited overrides. Script and argv may both be supplied.

**Postconditions:** Run `[cpptraj] + (argv or [])` without a shell, capture
stdout/stderr as text, and return stdout unchanged on exit code zero. Send the
script on stdin; append `"quit\n"` directly unless
`script.rstrip().lower().endswith("quit")`. With `script=None`, stdin input is
`None`. The caller's list/string is not mutated.

**Failures/limits:** Nonzero exit raises `subprocess.CalledProcessError` with
captured output; timeout and launch errors propagate. Successful exit does not
validate datasets or physical values. The suffix check is not a command parser;
no separating newline is inserted before appended `quit`. Tests should check
the exact subprocess stdin, argument list, environment, and failure propagation.

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

**Preconditions:** String or `Path` and a supported Zarr mode. Exact lowercase
suffix `.zip` selects an existing ZIP and requires `mode="r"`. Otherwise the
suffix is replaced with `.zarr`; stored paths are resolved. Use `r` for reads
and `w` for a new V2 working store; other writable modes are delegated to Zarr.

**Postconditions/invariants:** Expose `root`, `store`, `store_path`, and `mode`;
root is a group. Blocks created through `write` retain fixed item shape, dtype,
and axis-0 chunk length recorded in the three storage mappings. Direct root
mutation bypasses helper invariants. Storage does not enforce finite numbers,
physical units, identity maps, or agreement of semantic metadata with arrays.

**Checks/limits:** Invalid path types/non-read ZIP modes raise `ValueError`;
missing ZIP files raise `FileNotFoundError`. Opening/root errors propagate for
ZIP/read-only stores. In other modes, any opening/root error triggers a fallback
group creation/open with `mode="a"`, so the original error may not escape.
Construction calls `setdefault` for storage mappings even in read mode;
arbitrary external archives missing them are not guaranteed to open read-only.
`close()` suppresses store-closing errors. Context exit closes the store without
suppressing the body exception. No operation-after-close guarantee is specified.

### Blocks and attributes

| Public operation | Preconditions | Successful postconditions / explicit failures |
| --- | --- | --- |
| `write(these_arrays, to_block_named, *, arrays_per_chunk=None)` | Writable store; finite homogeneous items or stacked ndarray `(K, *item_shape)`; stable shape/dtype across appends; positive integer chunk length if supplied | Append `K` items in order, retaining old rows; return `None`. Empty input creates no block. Read mode raises `RuntimeError`; within-batch shape/dtype differences raise `ValueError`; cached shape/dtype/chunk conflicts raise `RuntimeError` |
| `read(from_block_named, ids=None)` | Existing array; valid axis-0 integer, slice, or 1D integer index collection | Detached NumPy copy with stored dtype. `None` reads all; an integer drops axis 0; slices/collections retain it. Collections use orthogonal indexing after NumPy index coercion. Missing name raises `KeyError`; non-array member raises `TypeError`; selection errors propagate |
| `block_iter(from_block_named, *, step=1)` | Existing array; positive integer step | Detached copies of consecutive slices, at most `step` rows each, covering axis 0 in order. Missing/non-array members raise `KeyError`/`TypeError`. Positive step is not explicitly validated; an externally created scalar array is yielded once |
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
`write` creates the block/metadata before validating remaining items, so failed
validation may leave an empty block. Resize precedes assignment: writes are
**not transactional**, and failed writes need not restore their prior state.

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
accepting a batch or index pair. For an empty sliceable input use a positive
size: the default/nonpositive branch otherwise attempts a zero range step and
raises `ValueError`.

**Postconditions:** Positive size yields consecutive nonempty batches of at most
that size, preserving each item once and in order. Nonpositive size yields one
batch for nonempty input. Range mode returns **0-based consumption positions**,
not input values: `(start, stop_exclusive)` or `(start, end_inclusive)`, wrapped
with `out_as`. Thus `batches_of([10, 20, 30], 2, ranges=True)` yields `[0, 2]`,
`[2, 3]` because the default wrapper is `list` (the docstring shows tuples).
`inclusive_end` does not affect data batches.

Empty input with positive size yields nothing. An empty generic iterator with
nonpositive size yields one empty batch, or `[0, 0]` / `[0, -1]` for exclusive /
inclusive ranges. Sliceable inputs are sliced; generic inputs are consumed into
lists, including in range mode. Nonpositive sizes provide no bounded-memory
guarantee. These index ranges are not CPPTRAJ's 1-based inclusive frame ranges.

### Small file/environment/pipeline helpers

| Helper | Preconditions | Postconditions / limits |
| --- | --- | --- |
| `read_lines(file_path, skip_header=True)` | Readable UTF-8 file | Lines retaining line endings, optionally removing exactly the first; empty file gives `[]`; file errors propagate |
| `temporary_file(prefix, suffix)` | Valid filename components; writable system temporary directory | `Path` to a new empty file with its handle already closed; caller owns deletion, including on failure |
| `create_updated_subprocess_env(**var_vals)` | String environment names; values convertible to strings (path-like values must produce strings for the annotated result) | Copy `os.environ`; remove keys with `None`; booleans become `"TRUE"`/`"FALSE"`, path-like values use `os.fspath`, others `str`; parent environment and omitted variables remain unchanged |
| `compose_steps(*steps)` | Each step is `(callable, kwargs_or_None)`; kwargs convertible by `dict`; compatible result/input types | Unary pipeline in supplied order; no steps returns input object unchanged; shallow-copy kwargs per call; propagate conversion/step errors; input mutation is allowed and a returned `None` becomes the next input |
| `current_time()` | None | Local wall-clock text `%Y-%m-%d_%H%M%S`; no timezone, uniqueness, or monotonicity guarantee |

`compose_steps` checks tuple length/callability at construction (`TypeError`),
but kwargs only at invocation. Copies are shallow: nested mutable values remain
shared. Legacy memory-based file/directory chunking, private V1 matrix/walk
internals, and neural backend implementations are outside this reuse scope.

## Verification boundary

Current tests in `tests/test_storage.py` cover basic stacked writes, chunk
defaults, within-batch mismatches, read-only deletion, and block listing.
They also cover context-manager cleanup and exception preservation, complete
archive publication, destination preservation on compression failures, temporary
output cleanup, and existing destination-path/ZIP-source behavior.
`tests/test_rin.py` checks synthetic parsing, RIN data flow, candidate
deduplication, and rejection of non-finite pairwise output. Walk/embedding tests
exercise ordered archives, seed forwarding, and warm starts; plotting fixtures
replace GUI setup. CPPTRAJ calls and pipeline shared memory are mocked.

These tests are not complete coverage of the contracts above. Before changing
reused code, add contract-based tests for its relevant success, boundary, and
failure cases, including preservation of existing public behavior. Real CPPTRAJ
extraction checks remain separate from the existing mocked suite; storage
failure tests do not establish power-loss durability or guaranteed cleanup when
the filesystem refuses removal.
