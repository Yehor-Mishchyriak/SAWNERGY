import os
import psutil
import numpy as np
import re
from ast import literal_eval
from math import ceil
from datetime import datetime
from concurrent.futures import as_completed
from shutil import which
from typing import Tuple, Callable, Iterable, List, Any, Optional, Dict

####################################################
# General functions distributed across the modules #
####################################################

def create_output_dir(output_directory_location: str, output_directory_name: str) -> str:
    """
    Creates a unique output directory by appending a timestamp to the base name.

    Args:
        output_directory_location (str): Path to an existing parent directory.
        output_directory_name (str): Base name for the new output directory.

    Returns:
        str: The full path to the created output directory in the form:
             output_directory_location/output_directory_name_{time}
    """
    current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    output_directory_path = os.path.join(output_directory_location, output_directory_name.format(time=current_time))
    os.makedirs(output_directory_path, exist_ok=True) # Note: it will not raise an error or override the "output_directory_path" if it already exists
    return output_directory_path

def process_elementwise(in_parallel: bool = False,
                        Executor: Optional[Callable[..., Any]] = None,
                        max_workers: Optional[int] = None,
                        capture_output: bool = True) -> Callable[[Iterable[Any], Callable[..., Any], Any], Optional[List[Any]]]:
    """
    Returns a function that processes elements from an iterable using a specified function,
    either sequentially or in parallel.

    Args:
        in_parallel (bool, optional): Whether to process elements in parallel. Defaults to False.
        Executor (Optional[Callable[..., Any]], optional): Executor class for parallel processing
            (e.g., ThreadPoolExecutor or ProcessPoolExecutor). Must be provided if in_parallel is True.
        max_workers (Optional[int], optional): Maximum number of concurrent workers. Defaults to None.
        capture_output (bool, optional): If True, collects and returns the results; otherwise, runs for side effects. Defaults to True.

    Returns:
        Callable: A function that takes an iterable, a function, and any extra arguments/keywords.
                  When called, it returns a list of results (if capture_output is True) or None.
    """
    def inner(iterable: Iterable[Any],
              function: Callable[..., Any],
              *extra_args: Any,
              **extra_kwargs: Any) -> Optional[List[Any]]:
        results: List[Any] = []
        if in_parallel:
            local_max_workers = max_workers if max_workers is not None else os.cpu_count()
            if Executor is None:
                raise ValueError("An 'Executor' argument must be provided if 'in_parallel' is True.")
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

def _file_chunks_generator(file_path: str, chunk_size: int, skip_header: bool = True) -> Iterable[List[str]]:
    """
    Generator that yields chunks of a file (as lists of lines) based on a specified chunk size in bytes.

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

def chunked_file(file_path: str, allowed_memory_percentage_hint: float, num_workers: int) -> Iterable[List[str]]:
    """
    Splits a file into chunks based on available memory per worker and returns a generator yielding those chunks.
    If the file size is smaller than the memory allocation per worker, all lines are returned at once.

    Args:
        file_path (str): Path to the file to be chunked.
        allowed_memory_percentage_hint (float): Fraction (between 0 and 1) of total available memory allocated per worker.
        num_workers (int): Number of workers.

    Returns:
        Iterable[List[str]]: A generator yielding chunks (lists of strings) from the file.

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
    yield from _file_chunks_generator(file_path, chunk_size)

def _dir_chunks_generator(file_paths: list[str], files_per_chunk: int, residual_files: int):
    """
    Yield chunks of file paths, distributing extra files to the first few chunks.

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
    Yield file path chunks from a directory based on memory limits per worker.

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

    # get all file paths in the dir (filtering out subdirs)
    file_paths = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file))]
    
    if not file_paths:
        raise ValueError(f"No files found in directory: {dir_path}")

    # use the first file's size as a representative for all files
    file_size = os.path.getsize(file_paths[0])

    # compute how many files can fit in the memory allotted per worker
    files_per_worker = memory_per_worker // file_size

    if files_per_worker < 1:
        raise MemoryError(f"The files contained in {dir_path} are too large. Cannot distribute the files across the workers. Solution: increase 'allowed_memory_percentage_hint', if possible, or decrease 'num_workers'")

    num_files = len(file_paths)
    files_per_chunk, residual_files = divmod(num_files, files_per_worker)

    yield from _dir_chunks_generator(file_paths, files_per_chunk, residual_files)
    
#############################
# Matrix related operations #
#############################

def _softmax(x: np.ndarray) -> np.ndarray:
    """
    Computes the numerically stable softmax of a vector.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        np.ndarray: Softmax-transformed vector.
    """
    max_x_i = np.max(x)
    exp_x = np.exp(x - max_x_i)
    return exp_x / np.sum(exp_x)

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalizes a vector using the softmax function.

    Args:
        v (np.ndarray): Input vector.

    Returns:
        np.ndarray: Softmax-normalized vector.
    """
    return _softmax(v)

def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """
    Normalizes each row of a matrix using the softmax function.

    Args:
        matrix (np.ndarray): Input matrix.

    Returns:
        np.ndarray: Row-normalized matrix.
    """
    return np.apply_along_axis(normalize_vector, axis=1, arr=matrix)

####################################
# MODULE-SPECIFIC HELPER FUNCTIONS #
####################################

###################
# frames_analyzer #
###################

def cpptraj_is_available_at(path: Optional[str] = None) -> Optional[str]:
    """
    Checks if the 'cpptraj' executable is available.

    Args:
        path (Optional[str], optional): Absolute path to the 'cpptraj' executable.
            If None or if the file at the given path is not executable, the system PATH is searched.
            Defaults to None.

    Returns:
        Optional[str]: The path to the 'cpptraj' executable if found, or None otherwise.
    """
    if path is None:
        return which("cpptraj")
    if os.path.isfile(path) and os.access(path, os.X_OK):
        return path
    return which("cpptraj")

def construct_batch_sequence(number_frames: int, batch_size: int) -> List[Tuple[int, int]]:
    """
    Constructs a sequence of frame batches based on the total number of frames and the batch size.

    Args:
        number_frames (int): Total number of frames.
        batch_size (int): Number of frames per batch.

    Returns:
        List[Tuple[int, int]]: A list of tuples where each tuple represents the start and end frame numbers for a batch.
    """
    number_batches, residual_frames = divmod(number_frames, batch_size)
    batches = [(batch_size * k - (batch_size - 1), batch_size * k) for k in range(1, number_batches + 1)]
    if residual_frames > 0:
        last_frame = batches[-1][1]
        residual_batch = (last_frame, last_frame + residual_frames)
        batches.append(residual_batch)
    return batches

######################
# analyses_processor #
######################

def _split_entry(res_atm: str) -> Tuple[str, str]:
    """
    Splits a residue-atom string into the residue name and index.

    Args:
        res_atm (str): Input string in the format "MOLECULE_INDEX@ATOM".

    Returns:
        Tuple[str, str]: A tuple containing the residue name and the index.

    Example:
        "TRP_91@C" -> ("TRP", "91")
    """
    # NOTE: for matching amino acids (AA) we would use r"([A-Z]{3})_...",
    # meaning that each AA abbreviation is exactly 3 capital letters in cpptraj output;
    # however, there are other molecules that we might want to capture that are not AAs,
    # for example nucleic acids; so we have the following regex string
    matched = re.search(r"([A-Z0-9]+)_(\d+)@[A-Z0-9]+", res_atm)
    try:
        residue = matched.group(1)
        index = matched.group(2)
    except AttributeError:
        raise ValueError(f"Cannot decompose the {res_atm} string due to a wrong format; Expected: MOLECULE_INDEX@ATOM")
    return residue, index

def _parse_cpptraj_electrostatics_line(line: str) -> str:
    """
    Parses a line of residue interaction data into a CSV-compatible format.

    Args:
        line (str): A line of interaction data.

    Returns:
        str: A CSV-formatted string with residue names, indices, and energy.
    """
    try:
        res_atm_A, _, _, res_atm_B, _, _, _, energy = line.split()
    except ValueError: # needed to add this line because cpptraj sometimes adds information on van der Waals (I think this is a bug)
        res_atm_A, _, _, res_atm_B, _, _, _, _, _, energy = line.split()
    res_A, res_A_index = _split_entry(res_atm_A)
    res_B, res_B_index = _split_entry(res_atm_B)
    return f"{res_A},{res_A_index},{res_B},{res_B_index},{energy}\n"

def read_lines(file_path: str, skip_header: bool = True) -> List[str]:
    """
    Reads all lines from a file, optionally skipping the header.

    Args:
        file_path (str): Path to the file.
        skip_header (bool, optional): Whether to skip the first line. Defaults to True.

    Returns:
        List[str]: A list of lines from the file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        return lines[1:] if skip_header else lines

def write_csv_header(header: str, output_file_path: str) -> None:
    """
    Writes a CSV header to the specified file.
    ## WARNING: This will override the file if it already exists.

    Args:
        header (str): The CSV header string.
        output_file_path (str): The file path where the header will be written.
    """
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write(header)

def write_csv_from_cpptraj_electrostatics(header: str, lines: List[str], output_file_path: str) -> None:
    """
    Writes parsed interaction data to a CSV file, including a header.

    Args:
        header (str): The CSV header string.
        lines (List[str]): A list of lines containing raw interaction data.
        output_file_path (str): The path to the output CSV file.
    """
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write(header)
        for line in lines:
            try:
                parsed_line = _parse_cpptraj_electrostatics_line(line)
            except Exception as e:
                raise ValueError(f"Error while parsing cpptraj files, wrong line format: {e}")
            output_file.write(parsed_line)

def append_csv_from_cpptraj_electrostatics(lines: List[str], output_file_path: str) -> None:
    """
    Appends parsed interaction data to an existing CSV file.

    Args:
        lines (List[str]): A list of lines containing raw interaction data.
        output_file_path (str): The path to the CSV file to which data will be appended.
    """
    with open(output_file_path, "a", encoding="utf-8") as output_file:
        for line in lines:
            try:
                parsed_line = _parse_cpptraj_electrostatics_line(line)
            except Exception as e:
                raise ValueError(f"Error while parsing cpptraj files, wrong line format: {e}")
            output_file.write(parsed_line)

#########################
# to_matrices_converter #
#########################

def frames_from_name(file_name: str) -> Tuple[int, int]:
    """
    Extracts the start and end frame numbers from a file name.

    Args:
        file_name (str): File name containing frame information in the format "start-end".

    Returns:
        Tuple[int, int]: The start and end frame numbers as integers.

    Raises:
        ValueError: If the file name does not contain two integers separated by a hyphen.
    """
    matched = re.search(r"(\d+)-(\d+)", file_name)
    try:
        start_frame = matched.group(1)
        end_frame = matched.group(2)
    except AttributeError:
        raise ValueError(f"Cannot decompose the {file_name} string due to a wrong format; Expected: int-int")
    return int(start_frame), int(end_frame)

###########
# protein #
###########

def _retrieve_matrices(matrices_directory_path: str, config: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the interaction and probability matrices stored as .npy files in a specified directory.

    This function uses the configuration employed during the network construction to determine the file names
    for the interactions and probabilities matrices. It scans the provided directory and loads the matrices
    using NumPy. If a matrix file is not found, the corresponding return value is None.

    Args:
        matrices_directory_path (str): The path to the directory containing the .npy matrix files.
        config (Dict[str, str]): A configuration file employed during the network construction.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple where the first element is the interaction matrix
        and the second element is the probability matrix.
    """
    config = config["ToMatricesConverter"]
    interaction_matrix = None
    probability_matrix = None
    for np_matrix in os.listdir(matrices_directory_path):
        np_matrix_path = os.path.join(matrices_directory_path, np_matrix)
        if np_matrix == config["interactions_matrix_name"]:
            interaction_matrix = np.load(np_matrix_path)
        elif np_matrix == config["probabilities_matrix_name"]:
            probability_matrix = np.load(np_matrix_path)
    return interaction_matrix, probability_matrix

def _retrieve_res_map(id_to_res_map_path: str) -> Tuple[str, ...]:
    """
    Retrieve the residue mapping from a file.

    The file is expected to contain a single tuple of residue names located at the first line.
    This function reads the file content and uses literal_eval to safely evaluate the string into a Python object.

    Args:
        id_to_res_map_path (str): The path to the file containing the residue mapping.

    Returns:
        Tuple[str, ...]: The residue mapping as a tuple.
    """
    with open(id_to_res_map_path, 'r') as file:
        id_to_res_string = file.read()
    # Note: literal_eval is safer than eval() because it only parses literals.
    id_to_res_map = literal_eval(id_to_res_string)
    return id_to_res_map

def import_network_components(directory_path: str, config: Dict[str, str]) -> Tuple[Tuple[str, ...], list[np.ndarray], list[np.ndarray]]:
    """
    Import network components from a directory based on the provided configuration.

    This function scans the specified directory for subdirectories and files. For each subdirectory, it retrieves the 
    interaction and probability matrices. If a file matching the residue mapping name (as specified 
    in the configuration) is found, it retrieves the residue mapping.
    
    Args:
        directory_path (str): The path to the directory containing the network component files.
        config (Dict[str, str]): The configuration file used for the network construction. Required for the directory scanning.

    Returns:
        Tuple[Tuple[str, ...], list[np.ndarray], list[np.ndarray]]:
            - The first element is the residue mapping (or None if not found).
            - The second element is a list of interaction matrices.
            - The third element is a list of probability matrices.
    """
    id_to_res_map: Tuple[str, ...] = None
    interaction_matrices: list = []
    probability_matrices: list = []
    sorted_paths = [os.path.join(directory_path, file) for file in sorted(os.listdir(directory_path))]
    for path_ in sorted_paths:
        if os.path.isdir(path_):
            interaction_matrix, probability_matrix = _retrieve_matrices(path_, config)
            interaction_matrices.append(interaction_matrix)
            probability_matrices.append(probability_matrix)
        elif os.path.basename(path_) == config["ToMatricesConverter"]["id_to_res_map_name"]:
            id_to_res_map = _retrieve_res_map(path_)
    
    return id_to_res_map, interaction_matrices, probability_matrices


if __name__ == "__main__":
    pass
