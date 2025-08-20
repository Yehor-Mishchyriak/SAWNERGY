import re
import logging
from math import ceil
from concurrent.futures import as_completed, ThreadPoolExecutor, ProcessPoolExecutor
from typing import Callable, Iterable, Any
import os, psutil
from pathlib import Path

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
#  PARALLEL PROCESSING AND EFFICIENT MEMORY USAGE RELATED FUNCTIONS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

def _apply(f: Callable, x: Any, extra_args: tuple, extra_kwargs: dict) -> Any:
    return f(x, *extra_args, **extra_kwargs)

def process_elementwise(
    in_parallel: bool = False,
    Executor: type[ThreadPoolExecutor] | type[ProcessPoolExecutor] | None = None,
    max_workers: int | None = None,
    capture_output: bool = True,
) -> Callable[[Iterable[Any], Callable[..., Any], Any], list[Any] | None]:
    
    def inner(iterable: Iterable[Any], function: Callable, *extra_args: Any, **extra_kwargs: Any) -> list[Any] | None:

        if not in_parallel:
            if capture_output:
                return [function(x, *extra_args, **extra_kwargs) for x in iterable]
            else:
                for x in iterable:
                    function(x, *extra_args, **extra_kwargs)
                return None

        if Executor is None:
            raise ValueError("An 'Executor' argument must be provided if 'in_parallel' is True.")

        local_max_workers = max_workers or (os.cpu_count() or 1)
        with Executor(max_workers=local_max_workers) as executor:
            futures = {executor.submit(_apply, function, x, extra_args, extra_kwargs): i
                       for i, x in enumerate(iterable)}
            if capture_output:
                results: list[Any] = [None] * len(futures)
                for fut in as_completed(futures):
                    idx = futures[fut]
                    results[idx] = fut.result()
                return results
            else:
                for fut in as_completed(futures):
                    fut.result()
                return None

    return inner

def files_from(dir_path: str, pattern: re.Pattern = None) -> list[str]:
    pattern = pattern or re.compile(r".*")
    dp = Path(dir_path)
    files = list()
    for file_name in sorted(os.listdir(dir_path)):
        pth = dp / file_name
        if pattern.match(file_name) and pth.is_file():
            files.append(str(pth))
    return files

def file_chunks_generator(file_path: str, chunk_size: int, skip_header: bool = True) -> Iterable[list[str]]:
    with open(file_path, "r", encoding="utf-8") as file:
        if skip_header:
            try:
                next(file)
            except StopIteration:
                return
        while True:
            chunk = file.readlines(chunk_size)
            if not chunk:
                break
            yield chunk

def chunked_file(file_path: str, allowed_memory_percentage_hint: float, num_workers: int) -> Iterable[list[str]]:
    if not (0 < allowed_memory_percentage_hint <= 1.0):
        raise ValueError(f"Invalid allowed_memory_percentage_hint parameter: expected a value between 0 and 1, instead got: {allowed_memory_percentage_hint}")
    
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")

    memory_per_worker = max(1, int((allowed_memory_percentage_hint * psutil.virtual_memory().available) / num_workers))

    file_size = os.path.getsize(file_path)
    if file_size <= memory_per_worker:
        yield read_lines(file_path)
        return

    num_chunks = max(1, ceil(file_size / memory_per_worker))
    chunk_size = max(1, file_size // num_chunks)

    yield from file_chunks_generator(file_path, chunk_size)

def dir_chunks_generator(file_paths: list[str], files_per_chunk: int, residual_files: int):
    total_files = len(file_paths)
    
    # if files_per_chunk is less than or equal to zero, then all the files fit into memory at once, so yield all of them
    if files_per_chunk <= 0:
        yield file_paths
        return
    
    # calculate the number of full chunks.
    # given: total_files = num_chunks * files_per_chunk + residual_files,
    # solve for num_chunks:
    num_chunks = (total_files - residual_files) // files_per_chunk
    
    start = 0
    for i in range(num_chunks):
        # distribute one extra file to the first 'residual_files' chunks
        chunk_size = files_per_chunk + 1 if i < residual_files else files_per_chunk
        yield file_paths[start:start + chunk_size]
        start += chunk_size

    # in case there are any leftover files because of rounding, yield them as a final chunk.
    if start < total_files:
        yield file_paths[start:]

def chunked_dir(dir_path: str, allowed_memory_percentage_hint: float, num_workers: int):
    if not (0 < allowed_memory_percentage_hint <= 1.0):
        raise ValueError(f"Invalid allowed_memory_percentage_hint parameter: expected a value between 0 and 1, instead got: {allowed_memory_percentage_hint}")
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    
    memory_per_worker = max(1, int((psutil.virtual_memory().available * allowed_memory_percentage_hint) / num_workers))

    file_paths = files_from(dir_path)
    if not file_paths:
        raise ValueError(f"No files found in directory: {dir_path}")

    file_size = os.path.getsize(file_paths[0])
    if file_size == 0:
        file_size = 1

    files_per_worker = int(memory_per_worker // file_size)

    if files_per_worker < 1:
        raise MemoryError(
            f"The files contained in {dir_path} are too large. Cannot distribute the files across the workers. "
            "Solution: increase 'allowed_memory_percentage_hint', if possible, or decrease 'num_workers'"
        )

    num_files = len(file_paths)

    files_per_chunk = files_per_worker
    residual_files = num_files % files_per_chunk

    yield from dir_chunks_generator(file_paths, files_per_chunk, residual_files)

def read_lines(file_path: str, skip_header: bool = True) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        return lines[1:] if (skip_header and lines) else lines


if __name__ == "__main__":
    pass
