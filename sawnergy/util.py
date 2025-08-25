from __future__ import annotations

# third-pary
import zarr
from zarr.storage import LocalStore, ZipStore
import numpy as np
# built-in
import re
import logging
from math import ceil
from concurrent.futures import as_completed, ThreadPoolExecutor, ProcessPoolExecutor
from typing import Callable, Iterable, Any
import os, psutil, tempfile
from pathlib import Path
import warnings


# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__)

# *----------------------------------------------------*
#                        CLASSES
# *----------------------------------------------------*

class ArrayStorage:
    def __init__(self, pth: Path | str) -> ArrayStorage:
        if isinstance(pth, str, Path): self.pth = Path(pth).resolve()
        else: raise ValueError(f"Expected 'str' or 'Path' input for 'pth' parameter; "
                               f"instead gotten: {type(pth)}")

        self.store = LocalStore(pth.with_suffix(".zarr"))
        self.root = zarr.group(store=self.store)

        self.arrays_per_chunk_in_block: dict[str, int] = dict()
        self.array_shape_in_block: dict[str, tuple[int, ...]] = dict()
        self.array_dtype_in_block: dict[str, np.dtype] = dict()

    # TO BE PROOFREAD AND DTYPE TO BE CACHED ALONG WITH LOGGING AND POTENTIAL WARNING
    def _setdefault(self,
                   named: str,
                   shape: tuple[int, ...],
                   dtype: np.dtype,
                   arrays_per_chunk: int | None = None):

        cached_shape = self.array_shape_in_block.get(named, None)
        if cached_shape is None:
            self.array_shape_in_block[named] = shape
        else:
            if not (cached_shape == shape):
                raise ValueError(f"Attempted to write an array of shape {shape} to the block of shape {cached_shape}")
        
        if arrays_per_chunk is None:
            arrays_per_chunk = self.arrays_per_chunk_in_block.get(named, None)
            if arrays_per_chunk is None:
                arrays_per_chunk = 10
                warnings.warn(f"You never set 'arrays_per_chunk' value for the block named {named}. "
                              f"So the default value of '10' was used, which may be suboptimal "
                              f"in your situation and cause worse performance. "
                              f"Considering array sizes in the block, choose a value "
                              f"given the amount of available RAM on your computer", RuntimeWarning)
                _logger.warning(f"'arrays_per_chunk' was never provided for the {named} block")
        else:
            self.arrays_per_chunk_in_block[named] = arrays_per_chunk
        
        return self.root.require_array(
            name=named,
            shape=(0,) + shape, # grow along axis 0
            chunks=(arrays_per_chunk,) + shape, # store as chunks of 'arrays_per_chunk' along axis 0
            dtype=dtype
        )

    # TO BE PROOFREAD
    def write(self, *,
              these_arrays: list[np.ndarray],
              to_block_named: str,
              arrays_per_chunk: int | None = None) -> None:
        if not these_arrays:
            return
        arr_example = these_arrays[0]
        block = self._setdefault(to_block_named, arr_example.shape, arr_example.dtype, arrays_per_chunk)
        data = np.asarray(these_arrays, dtype=arr_example.dtype); k = len(these_arrays)
        start = block.shape[0] # shape along the growable axis
        block.resize((start + k,) + arr_example.shape)
        block[start:start + k, ...] = data

    # TO BE COMPLETED
    def delete_block(self, named):
        pass

    # TO BE COMPLETED
    def compress(self) -> str:
        zip_path = self.pth.with_suffix(".zip")
        with ZipStore(zip_path, mode="w") as z:
            src_root = self.root
            dst_root = zarr.group(store=z)
            for key in src_root.keys():
                ...
                
                
                
                src = src_root[key]
                if isinstance(src, zarr.Array):
                    dst = dst_root.create_array(
                        name=key, shape=src.shape, chunks=src.chunks, dtype=src.dtype
                    )
                    # stream copy by chunks along axis 0
                    step = src.chunks[0] or max(1, src.shape[0])
                    for i in range(0, src.shape[0], step):
                        j = min(i + step, src.shape[0])
                        dst[i:j, ...] = src[i:j, ...]
                else:
                    # if you nest groups, add a small recursive copier
                    dst_root.create_group(key)
        return zip_path

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
#  PARALLEL PROCESSING AND EFFICIENT MEMORY USAGE RELATED FUNCTIONS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

def _apply(f: Callable, x: Any, extra_args: tuple, extra_kwargs: dict) -> Any:
    return f(x, *extra_args, **extra_kwargs)

def elementwise_processor(
    in_parallel: bool = False,
    Executor: type[ThreadPoolExecutor] | type[ProcessPoolExecutor] | None = None,
    max_workers: int | None = None,
    capture_output: bool = True,
) -> Callable[[Iterable[Any], Callable[..., Any], Any], list[Any] | None]:
    """Factory that returns a function to process an iterable elementwise.

    The returned callable executes a provided `function` over each element of an
    `iterable`, either sequentially or in parallel using the specified
    `Executor`. Results are optionally captured and returned as a list.

    Args:
        in_parallel: If True, process with a concurrent executor; otherwise run sequentially.
        Executor: Executor class to use when `in_parallel` is True
            (e.g., `ThreadPoolExecutor` or `ProcessPoolExecutor`). Ignored if `in_parallel` is False.
        max_workers: Maximum parallel workers. Defaults to `os.cpu_count()` when None.
        capture_output: If True, collect and return results; if False, execute for side effects and return None.

    Returns:
        A callable with signature:
            `(iterable, function, *extra_args, **extra_kwargs) -> list | None`
        When `capture_output` is True, the list preserves the input order.

    Raises:
        ValueError: If `in_parallel` is True and `Executor` is None.
        Exception: Any exception raised by `function` for a given element is propagated.

    Notes:
        - In parallel mode, task results are re-ordered to match input order.
        - In non-capturing modes, tasks are still awaited so exceptions surface.

    Example:
        >>> runner = elementwise_processor(in_parallel=True, Executor=ThreadPoolExecutor, max_workers=4)
        >>> out = runner(range(5), lambda x: x * 2)
        >>> out
        [0, 2, 4, 6, 8]
    """
    def inner(iterable: Iterable[Any], function: Callable, *extra_args: Any, **extra_kwargs: Any) -> list[Any] | None:
        """Execute `function` over `iterable` per the configuration of the factory.

        Args:
            iterable: Collection of input elements to process.
            function: Callable applied to each element of `iterable`.
            *extra_args: Extra positional arguments forwarded to `function`.
            **extra_kwargs: Extra keyword arguments forwarded to `function`.

        Returns:
            List of results when `capture_output` is True; otherwise None.

        Raises:
            ValueError: If `Executor` is missing while `in_parallel` is True.
            Exception: Any exception raised by `function` is propagated.
        """
        _logger.debug(
            "elementwise_processor: in_parallel=%s, Executor=%s, max_workers=%s, capture_output=%s, func=%s",
            in_parallel, getattr(Executor, "__name__", None), max_workers, capture_output, getattr(function, "__name__", repr(function))
        )

        if not in_parallel:
            _logger.info("elementwise_processor: running sequentially")
            if capture_output:
                result = [function(x, *extra_args, **extra_kwargs) for x in iterable]
                _logger.info("elementwise_processor: sequential completed with %d results", len(result))
                return result
            else:
                for x in iterable:
                    function(x, *extra_args, **extra_kwargs)
                _logger.info("elementwise_processor: sequential completed (no capture)")
                return None

        if Executor is None:
            _logger.error("elementwise_processor: Executor is required when in_parallel=True")
            raise ValueError("An 'Executor' argument must be provided if 'in_parallel' is True.")

        local_max_workers = max_workers or (os.cpu_count() or 1)
        _logger.info("elementwise_processor: starting parallel with %d workers via %s", local_max_workers, Executor.__name__)
        with Executor(max_workers=local_max_workers) as executor:
            futures = {executor.submit(_apply, function, x, extra_args, extra_kwargs): i
                       for i, x in enumerate(iterable)}
            _logger.info("elementwise_processor: submitted %d tasks", len(futures))
            if capture_output:
                results: list[Any] = [None] * len(futures)
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        results[idx] = fut.result()
                    except Exception:
                        _logger.exception("elementwise_processor: task %d raised", idx)
                        raise
                _logger.info("elementwise_processor: parallel completed with %d results", len(results))
                return results
            else:
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception:
                        _logger.exception("elementwise_processor: task %d raised", futures[fut])
                        raise
                _logger.info("elementwise_processor: parallel completed (no capture)")
                return None

    return inner

def files_from(dir_path: str, pattern: re.Pattern = None) -> list[str]:
    """List files in a directory matching a regex pattern.

    Args:
        dir_path: Path to the directory to scan.
        pattern: Compiled regex pattern to match file names. If None, matches all files.

    Returns:
        A sorted list of absolute (string) file paths present in `dir_path` that match `pattern`.

    Notes:
        - Only regular files are returned; directories are ignored.
        - Raises `FileNotFoundError`/`PermissionError` if `dir_path` is invalid/inaccessible.
    """
    pat = pattern or re.compile(r".*")
    dp = Path(dir_path)
    _logger.debug("files_from: scanning %s with pattern=%r", dp, pat.pattern if hasattr(pat, "pattern") else pat)
    files = list()
    for file_name in sorted(os.listdir(dir_path)):
        pth = dp / file_name
        if pat.match(file_name) and pth.is_file():
            files.append(str(pth))
    _logger.debug("files_from: matched %d files in %s", len(files), dp)
    return files

def file_chunks_generator(file_path: str, chunk_size: int, skip_header: bool = True) -> Iterable[list[str]]:
    """Yield lists of text lines from a file using a size-hint per chunk.

    Uses `io.TextIOBase.readlines(sizehint)` to read approximately `chunk_size`
    bytes per iteration, always ending on a line boundary.

    Args:
        file_path: UTF-8 encoded text file to read.
        chunk_size: Approximate number of bytes to read per chunk (size hint).
        skip_header: If True, skip the first line before yielding content.

    Yields:
        Lists of strings, each list containing complete lines.

    Notes:
        - The `sizehint` is approximate; chunks may be larger or smaller.
        - If `skip_header` is True and the file is empty, the generator returns immediately.
    """
    _logger.info("file_chunks_generator: file=%s chunk_size=%d skip_header=%s", file_path, chunk_size, skip_header)
    with open(file_path, "r", encoding="utf-8") as file:
        if skip_header:
            try:
                next(file)
                _logger.debug("file_chunks_generator: skipped header line")
            except StopIteration:
                _logger.info("file_chunks_generator: file empty after header skip")
                return
        while True:
            chunk = file.readlines(chunk_size)
            if not chunk:
                break
            yield chunk
    _logger.debug("file_chunks_generator: completed for %s", file_path)

def chunked_file(file_path: str, allowed_memory_percentage_hint: float, num_workers: int) -> Iterable[list[str]]:
    """Split a file into line chunks sized by per-worker memory allowance.

    Heuristically plans chunk sizes from available system memory and the
    declared number of workers, then yields line lists produced by
    `file_chunks_generator`.

    Args:
        file_path: Path to a UTF-8 text file.
        allowed_memory_percentage_hint: Fraction in (0, 1] of *available* RAM to budget in total,
            divided across workers.
        num_workers: Number of workers the chunks are intended for.

    Yields:
        Lists of strings representing line chunks of the file.

    Raises:
        ValueError: If `allowed_memory_percentage_hint` not in (0, 1] or `num_workers` < 1.

    Notes:
        - This is a heuristic: Python string overhead and decoding expand beyond raw bytes.
        - If the whole file fits within `memory_per_worker`, a single chunk is yielded.
    """
    if not (0 < allowed_memory_percentage_hint <= 1.0):
        _logger.error("chunked_file: invalid allowed_memory_percentage_hint=%s", allowed_memory_percentage_hint)
        raise ValueError(f"Invalid allowed_memory_percentage_hint parameter: expected a value between 0 and 1, instead got: {allowed_memory_percentage_hint}")
    
    if num_workers < 1:
        _logger.error("chunked_file: num_workers must be >= 1 (got %s)", num_workers)
        raise ValueError("num_workers must be at least 1")

    memory_per_worker = max(1, int((allowed_memory_percentage_hint * psutil.virtual_memory().available) / num_workers))
    file_size = os.path.getsize(file_path)
    _logger.info("chunked_file: file_size=%d bytes, memory_per_worker=%d bytes", file_size, memory_per_worker)

    if file_size <= memory_per_worker:
        _logger.info("chunked_file: file fits in memory per worker; yielding all lines")
        yield read_lines(file_path)
        return

    num_chunks = max(1, ceil(file_size / memory_per_worker))
    chunk_size = max(1, file_size // num_chunks)
    _logger.info("chunked_file: planning %d chunks (~%d bytes each)", num_chunks, chunk_size)

    yield from file_chunks_generator(file_path, chunk_size)

def dir_chunks_generator(file_paths: list[str], files_per_chunk: int, residual_files: int):
    """Yield lists of file paths partitioned by a base chunk size and residuals.

    Distributes `residual_files` by giving the first `residual_files` chunks one
    extra file each.

    Args:
        file_paths: Full list of file paths to chunk.
        files_per_chunk: Base number of files to include in each chunk (>= 0).
        residual_files: Number of initial chunks that should receive one additional file.

    Yields:
        Slices (lists) of `file_paths` representing each chunk.

    Notes:
        - If `files_per_chunk <= 0`, all files are yielded as a single chunk.
        - The final tail (if any) is yielded after full chunks.
    """
    total_files = len(file_paths)
    _logger.debug("dir_chunks_generator: total_files=%d, files_per_chunk=%d, residual_files=%d",
                  total_files, files_per_chunk, residual_files)
    
    if files_per_chunk <= 0:
        _logger.info("dir_chunks_generator: files_per_chunk<=0 → yielding all %d files at once", total_files)
        yield file_paths
        return
    
    num_chunks = (total_files - residual_files) // files_per_chunk
    _logger.debug("dir_chunks_generator: full_chunks=%d", num_chunks)
    
    start = 0
    for i in range(num_chunks):
        chunk_size = files_per_chunk + 1 if i < residual_files else files_per_chunk
        yield file_paths[start:start + chunk_size]
        start += chunk_size

    if start < total_files:
        _logger.debug("dir_chunks_generator: yielding tail chunk of %d files", total_files - start)
        yield file_paths[start:]

def chunked_dir(dir_path: str, allowed_memory_percentage_hint: float, num_workers: int):
    """Plan directory file chunks to fit a per-worker memory hint.

    Assumes files in `dir_path` are of similar size (uses the first file as a
    representative) to estimate how many files can be processed per worker.
    Yields lists of file paths sized accordingly.

    Args:
        dir_path: Directory containing the files to chunk.
        allowed_memory_percentage_hint: Fraction in (0, 1] of available RAM to allocate across workers.
        num_workers: Number of workers that will process the chunks.

    Yields:
        Lists of file paths sized for concurrent processing.

    Raises:
        ValueError: If inputs are invalid or the directory is empty.
        MemoryError: If a single file is too large for the per-worker memory allowance.

    Notes:
        - If the first file is empty, a 1-byte surrogate is used to avoid division by zero.
        - Actual memory usage depends on file content and processing overhead.
    """
    if not (0 < allowed_memory_percentage_hint <= 1.0):
        _logger.error("chunked_dir: invalid allowed_memory_percentage_hint=%s", allowed_memory_percentage_hint)
        raise ValueError(f"Invalid allowed_memory_percentage_hint parameter: expected a value between 0 and 1, instead got: {allowed_memory_percentage_hint}")
    if num_workers < 1:
        _logger.error("chunked_dir: num_workers must be >= 1 (got %s)", num_workers)
        raise ValueError("num_workers must be at least 1")
    
    memory_per_worker = max(1, int((psutil.virtual_memory().available * allowed_memory_percentage_hint) / num_workers))
    _logger.info("chunked_dir: memory_per_worker=%d bytes (hint=%s, workers=%d)", memory_per_worker, allowed_memory_percentage_hint, num_workers)

    file_paths = files_from(dir_path)
    if not file_paths:
        _logger.error("chunked_dir: no files found in directory %s", dir_path)
        raise ValueError(f"No files found in directory: {dir_path}")

    file_size = os.path.getsize(file_paths[0])
    if file_size == 0:
        _logger.warning("chunked_dir: first file is zero bytes; falling back to 1 byte for sizing")
        file_size = 1

    files_per_worker = int(memory_per_worker // file_size)
    _logger.info("chunked_dir: representative_file_size=%d bytes -> files_per_worker=%d", file_size, files_per_worker)

    if files_per_worker < 1:
        _logger.error("chunked_dir: files too large for current memory hint per worker")
        raise MemoryError(
            f"The files contained in {dir_path} are too large. Cannot distribute the files across the workers. "
            "Solution: increase 'allowed_memory_percentage_hint', if possible, or decrease 'num_workers'"
        )

    num_files = len(file_paths)
    files_per_chunk = files_per_worker
    residual_files = num_files % files_per_chunk
    _logger.info("chunked_dir: num_files=%d -> files_per_chunk=%d, residual_files=%d", num_files, files_per_chunk, residual_files)

    yield from dir_chunks_generator(file_paths, files_per_chunk, residual_files)

def read_lines(file_path: str, skip_header: bool = True) -> list[str]:
    """Read all lines from a UTF-8 text file, optionally skipping the header.

    Args:
        file_path: Path to the input file.
        skip_header: If True, omit the first line from the returned list.

    Returns:
        A list of lines (strings). If `skip_header` is True and the file is
        non-empty, the first line is excluded.

    Notes:
        - Uses `readlines()`; for gigantic files prefer streaming approaches.
    """
    _logger.debug("read_lines: reading %s (skip_header=%s)", file_path, skip_header)
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        _logger.debug("read_lines: read %d lines from %s", len(lines), file_path)
        return lines[1:] if (skip_header and lines) else lines

def temporary_file(prefix: str, suffix: str) -> Path:
    """Create a named temporary file and return its path.

    This helper creates a `NamedTemporaryFile`, closes it immediately, and
    returns its filesystem path so other processes can open/write it later.
    The caller is responsible for deleting the file when finished.

    Args:
      prefix: Filename prefix used when creating the temporary file.
      suffix: Filename suffix (e.g., extension) used when creating the file.

    Returns:
      Path: Filesystem path to the created temporary file.

    Notes:
      The file is created on the default temporary directory for the system.
      The file handle is closed before returning, so only the path is kept.
    """
    ntf = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete="False")
    ntf.close()
    return Path(ntf.name)

def batches_of(iterable: Iterable,
               batch_size: int = -1,
               *,
               out_as: type = list,
               ranges: bool = False,
               inclusive_end: bool = False):
    """Yield elements of `iterable` in fixed-size batches or index ranges.

    This function requires that `iterable` supports `len()` and slicing.
    When `ranges=True`, yields index pairs instead of slices.

    Args:
      iterable: A sequence-like object supporting `len()` and slicing.
      batch_size: Number of items per batch. If <= 0, the entire iterable is
        yielded in a single batch. Defaults to -1.
      out_as: Constructor used to wrap each yielded batch (e.g., `list`, `tuple`)
        or to wrap the index pair when `ranges=True`. Defaults to `list`.
      ranges: If True, yield index ranges instead of actual data slices.
        Each yielded item is `(start, end)` (exclusive) unless `inclusive_end`
        is True. Defaults to False.
      inclusive_end: If `ranges=True`, control whether the returned range end
        index is inclusive (`(start, end_inclusive)`) or exclusive
        (`(start, end_exclusive)`). Ignored when `ranges=False`. Defaults to False.

    Yields:
      Any: For `ranges=False`, a batch containing up to `batch_size` elements,
      wrapped with `out_as`. For `ranges=True`, an index pair `(start, end)` (or
      `(start, end_inclusive)` if `inclusive_end=True`) wrapped with `out_as`.

    Raises:
      TypeError: If `iterable` does not support `len()` or slicing.

    Examples:
      Yield data batches:

      >>> list(batches_of([1,2,3,4,5], batch_size=2))
      [[1, 2], [3, 4], [5]]

      Yield index ranges (exclusive end):

      >>> list(batches_of(range(10), batch_size=4, ranges=True))
      [[0, 4], [4, 8], [8, 10]]
    """
    n = len(iterable)
    if batch_size <= 0:
        batch_size = n
    for start in range(0, n, batch_size):
        end_excl = min(start + batch_size, n)
        if ranges:
            if inclusive_end:
                yield out_as((start, end_excl - 1))
            else:
                yield out_as((start, end_excl))
        else:
            yield out_as(iterable[start:end_excl])

def create_updated_subprocess_env(**var_vals: Any) -> dict[str, str]:
    """Return a copy of the current environment with specified overrides.

    Convenience helper for preparing an `env` dict to pass to `subprocess.run`.
    Values are converted to strings; booleans map to ``"TRUE"``/``"FALSE"``.
    If a value is `None`, the variable is removed from the child environment.
    Path-like values are converted via `os.fspath`.

    Args:
      **var_vals: Mapping of environment variable names to desired values.
        - `None`: remove the variable from the environment.
        - `bool`: stored as `"TRUE"` or `"FALSE"`.
        - `int`, `str`, path-like: converted to `str` (path-like via `os.fspath`).

    Returns:
      dict[str, str]: A new environment dictionary suitable for `subprocess.run`.

    Examples:
      >>> env = create_updated_subprocess_env(OMP_NUM_THREADS=1, MKL_DYNAMIC=False)
      >>> env["OMP_NUM_THREADS"]
      '1'
      >>> env["MKL_DYNAMIC"]
      'FALSE'
    """
    env: dict[str, str] = os.environ.copy()
    for var, val in var_vals.items():
        if val is None:
            env.pop(var, None)
        elif isinstance(val, bool):
            env[var] = "TRUE" if val else "FALSE"
        else:
            env[var] = os.fspath(val) if hasattr(val, "__fspath__") else str(val)
    return env


if __name__ == "__main__":
    pass
