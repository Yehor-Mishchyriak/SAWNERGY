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

def process_elementwise(in_parallel: bool = False,
                        Executor: type[ThreadPoolExecutor] | type[ProcessPoolExecutor] | None = None,
                        max_workers: int | None = None,
                        capture_output: bool = True) -> Callable:
    def inner(iterable: Iterable[Any],
              function: Callable,
              *extra_args: Any,
              **extra_kwargs: Any) -> list[Any] | None:
        results = []
        if in_parallel:
            if Executor is None:
                raise ValueError("An 'Executor' argument must be provided if 'in_parallel' is True.")
            local_max_workers = max_workers or os.cpu_count()
            with Executor(max_workers=local_max_workers) as executor:
                tasks = [executor.submit(function, element, *extra_args, **extra_kwargs) for element in iterable]
                for future in as_completed(tasks):
                    if capture_output:
                        results.append(future.result())
                    else:
                        future.result()
            return results if capture_output else None
        else:
            return [function(element, *extra_args, **extra_kwargs) for element in iterable]
    return inner

def files_from(dir_path: str, pattern: re.Pattern = None) -> list[str]:
    pattern = pattern or re.compile(r".*")
    files = list()
    for file_name in sorted(os.listdir(dir_path)):
        pth = Path(dir_path) / file_name
        if re.match(pattern, file_name) and pth.is_file():
            files.append(str(pth))
    return files

def file_chunks_generator(file_path: str, chunk_size: int, skip_header: bool = True) -> Iterable[list[str]]:
    """
    Generator that yields chunks of a UTF-8 encoded file (as lists of lines) based on a specified chunk size in bytes.

    Args:
        file_path (str): Path to the file to be read.
        chunk_size (int): Approximate maximum number of bytes per chunk.
        skip_header (bool, optional): Whether to skip the first line of the file. Defaults to True.

    Yields:
        List[str]: A list of complete lines from the file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        if skip_header:
            next(file)
        while True:
            chunk = file.readlines(chunk_size)
            if not chunk:
                break
            yield chunk

def chunked_file(file_path: str, allowed_memory_percentage_hint: float, num_workers: int) -> Iterable[list[str]]:
    """
    Splits a UTF-8 encoded file into chunks based on available memory per worker and returns a generator yielding those chunks.
    If the file size is smaller than the memory allocation per worker, all lines are returned at once.

    Args:
        file_path (str): Path to the file to be chunked.
        allowed_memory_percentage_hint (float): Fraction (between 0 and 1) of total available memory allocated per worker.
        num_workers (int): Number of workers.

    Returns:
        Iterable[list[str]]: A generator yielding chunks (lists of strings) from the file.

    Raises:
        ValueError: If allowed_memory_percentage_hint is not between 0 and 1.
    """
    if not (0 < allowed_memory_percentage_hint <= 1.0):
        raise ValueError(f"Invalid allowed_memory_percentage_hint parameter: expected a value between 0 and 1, instead got: {allowed_memory_percentage_hint}")
    
    # ensure num_workers is at least 1.
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")

    memory_per_worker = (allowed_memory_percentage_hint * psutil.virtual_memory().available) / num_workers
    file_size = os.path.getsize(file_path)
    if file_size <= memory_per_worker:
        yield read_lines(file_path)
        return
    num_chunks = ceil(file_size / memory_per_worker)
    chunk_size = file_size // num_chunks
    yield from file_chunks_generator(file_path, chunk_size)

def dir_chunks_generator(file_paths: list[str], files_per_chunk: int, residual_files: int):
    """
    Yield lists of file paths, distributing 'residual_files' to the first few chunks.

    Parameters:
        file_paths (list[str]): List of file paths.
        files_per_chunk (int): Base number of files per chunk.
        residual_files (int): Number of chunks that get one extra file.

    Yields:
        list[str]: A chunk of file paths.
    """
    total_files = len(file_paths)
    
    # if files_per_chunk is zero, then all the files fit into memory at once, so yield all of them
    if files_per_chunk == 0:
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
    """
    Yield lists of file paths from a directory based on memory limits per worker.

    Parameters:
        dir_path (str): Directory containing files.
        allowed_memory_percentage_hint (float): Fraction of available memory to allocate per worker (0 < value <= 1).
        num_workers (int): Number of workers to process the files.

    Yields:
        list[str]: A chunk of file paths sized for concurrent processing.

    Raises:
        ValueError: For invalid parameters or if no files are found.
        MemoryError: If a single file exceeds the memory per worker.
    """
    # validate the memory hint
    if not (0 < allowed_memory_percentage_hint <= 1.0):
        raise ValueError(f"Invalid allowed_memory_percentage_hint parameter: expected a value between 0 and 1, instead got: {allowed_memory_percentage_hint}")
    
    # ensure num_workers is at least 1
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    
    # compute the memory allowed per worker.
    memory_per_worker = (psutil.virtual_memory().available * allowed_memory_percentage_hint) / num_workers

    # get all file paths in the dir
    file_paths = files_from(dir_path)
    
    if not file_paths:
        raise ValueError(f"No files found in directory: {dir_path}")

    # use the first file's size as a representative for all files
    file_size = os.path.getsize(file_paths[0])

    # compute how many files can fit in the memory allotted per worker
    files_per_worker = memory_per_worker // file_size

    if files_per_worker < 1:
        raise MemoryError(f"The files contained in {dir_path} are too large. Cannot distribute the files across the workers. " 
                          "Solution: increase 'allowed_memory_percentage_hint', if possible, or decrease 'num_workers'")

    num_files = len(file_paths)
    files_per_chunk, residual_files = divmod(num_files, files_per_worker)

    yield from dir_chunks_generator(file_paths, files_per_chunk, residual_files)

def read_lines(file_path: str, skip_header: bool = True) -> list[str]:
    """
    Reads all lines from a UTF-8 encoded file, optionally skipping the header.

    Args:
        file_path (str): Path to the file.
        skip_header (bool, optional): Whether to skip the first line. Defaults to True.

    Returns:
        list[str]: A list of lines from the file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        return lines[1:] if skip_header else lines


if __name__ == "__main__":
    pass
