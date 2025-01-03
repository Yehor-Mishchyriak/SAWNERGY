#!AllostericPathwayAnalyzer/venv/bin/python3

import os
import atexit
import logging
from logging.config import dictConfig
from datetime import datetime
import numpy as np
from json import load
from concurrent.futures import as_completed
from re import search
from functools import wraps
import ast

####################################################
# General functions distributed across the modules #
####################################################

def load_json_config(config_location: str) -> dict:
    """
    Loads a JSON configuration file.

    Args:
        config_location (str): Path to the JSON configuration file.

    Returns:
        dict: Parsed JSON configuration.
    """
    with open(config_location, "r") as config_file:
        config = load(config_file)
    return config

def create_output_dir(output_directory_location: str, output_directory_name: str) -> str:
    """
    Creates a unique output directory.

    Args:
        output_directory_location (str): Path to the parent directory.
        output_directory_name (str): Base name for the output directory.

    Returns:
        str: Path to the created output directory.
    """
    # get the current time to ensure that the name of the output is unique
    current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

    # create the complete path to the output directory
    output_directory_path = os.path.join(output_directory_location, f"{output_directory_name}_{current_time}")

    # create the output directory
    os.makedirs(output_directory_path, exist_ok=True)

    return output_directory_path

def set_up_logging(config_location, logger_name, handler_name):
    """
    Configures logging based on a JSON configuration file.

    Args:
        config_location (str): Path to the logging configuration file.
        logger_name (str): Name of the logger to set up.
        handler_name (str): Name of the logging handler to use.

    Returns:
        logging.Logger: Configured logger.
    """
    try:
        # load and configure logging
        config = load_json_config(config_location)
        dictConfig(config)
    except Exception as e:
        raise RuntimeError(f"Failed to load or configure logging: {e}")

    # retrieve the handler by name
    handler = logging.getHandlerByName(handler_name)
    if handler is None:
        raise ValueError(f"Handler '{handler_name}' not found in the logging configuration.")

    # start the listener if the handler supports it
    if hasattr(handler, 'listener'):
        handler.listener.start()
        atexit.register(handler.listener.stop)

    # return the configured logger
    return logging.getLogger(logger_name)

def process_elementwise(in_parallel=False, Executor=None, max_workers=None):
    """
    Processes elements in an iterable using a function, either sequentially or in parallel.

    Args:
        in_parallel (bool, optional): Whether to process in parallel. Defaults to False.
        Executor: Executor class for parallel processing (e.g., ThreadPoolExecutor).
        max_workers (int, optional): Maximum number of workers for parallel processing.

    Returns:
        function: A function to process elements from an iterable.
    """
    def inner(iterable, function, *extra_args, **extra_kwargs):

        nonlocal in_parallel
        nonlocal Executor
        nonlocal max_workers

        results = []
        if in_parallel:

            if max_workers is None:
                max_workers = os.cpu_count()
            
            if Executor is None:
                raise ValueError("An 'Executor' argument must be provided.")

            with Executor(max_workers=max_workers) as executor:
                tasks = [executor.submit(function, element, *extra_args, **extra_kwargs) for element in iterable]

                for future in as_completed(tasks):
                    result = future.result()
                    results.append(result)
        else:
            results = [function(element, *extra_args, **extra_kwargs) for element in iterable]
        return results
    
    return inner

# DECORATORS FOR GENERIC ERROR HANDLING AND LOGGING:

def init_error_handler_n_logger(logger):
    """
    A decorator to log exceptions occurring during __init__.

    Args:
        logger: A logging.Logger instance used for logging exceptions.
    Returns:
        A wrapped function with error logging capabilities.
    """
    def decorator(init):
        @wraps(init)
        def wrapper(*args, **kwargs):
            try:
                # attempt to initialize the class
                return init(*args, **kwargs)
            except OSError as e:
                class_name = args[0].__class__.__name__
                logger.critical(
                    f"Failed to create or access the output directory during {class_name} initialization. "
                    f"Error: {e}"
                )
                raise OSError(
                    f"An error occurred while creating or accessing the output directory for {class_name} class: {e}. "
                    f"Check permissions and available disk space."
                ) from e
            except KeyError as e:
                class_name = args[0].__class__.__name__
                logger.critical(f"Corrupt/incompatible root configuration file in {class_name}: {e}")
                raise KeyError(f"Missing configuration key: {e}") from e
            except Exception as e:
                class_name = args[0].__class__.__name__
                logger.exception(f"Unexpected error occurred during {class_name} initialization: {e}")
                raise
        return wrapper
    return decorator

def generic_error_handler_n_logger(logger, exclude_logging_exceptions=()):
    """
    A decorator to log unexpected exceptions during the execution of a function.

    Args:
        logger: A logging.Logger instance used for logging exceptions.
        exclude_exceptions (tuple): A tuple of exception types to exclude from logging.

    Returns:
        A wrapped function with error logging capabilities.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exclude_logging_exceptions as e:
                # skip logging for excluded exceptions, just re-raise
                raise
            except Exception as e:
                # log unexpected exceptions
                func_name = func.__name__
                logger.exception(f"Unexpected error occurred in function '{func_name}': {e}")
                raise
        return wrapper
    return decorator

#############################
# Matrix related operations #
#############################

# numerically stable softmax (no overflow even for large values of x_i)
def _softmax(x: np.array):
    """
    Computes the numerically stable softmax of a vector.

    Args:
        x (np.array): Input vector.

    Returns:
        np.array: Softmax-transformed vector.
    """
    # x is a vector
    max_x_i = np.max(x)
    # elementwise exponentiation for vector x
    exp_x = np.exp(x - max_x_i) # (note the trick with the uniform subtraction of the largest entry)
    return exp_x / np.sum(exp_x) # normalize each entry in the exponent space

# numerically stable log-softmax: avoiding overflow with large exponents
# as well as avoiding underflow when multiplying small probabilities
# note the necessity of exponentiating eventually for converting back from the log space
def _log_softmax(x: np.array):
    """
    Computes the numerically stable log-softmax of a vector.

    Args:
        x (np.array): Input vector.

    Returns:
        np.array: Log-softmax-transformed vector.
    """
    # x is a vector
    max_x_i = np.max(x)
    # add up the logs of sums
    log_sum_exp = max_x_i + np.log(np.sum(np.exp(x - max_x_i)))
    return x - log_sum_exp # normalize in the log space

def normalize_vector(v: np.array):
    """
    Normalizes a vector using the softmax function.

    Args:
        v (np.array): Input vector.

    Returns:
        np.array: Softmax-normalized vector.
    """
    return _softmax(v)

def normalize_rows(matrix: np.array):
    """
    Normalizes each row of a matrix using the softmax function.

    Args:
        matrix (np.array): Input matrix.

    Returns:
        np.array: Row-normalized matrix.
    """
    return np.apply_along_axis(normalize_vector, axis=1, arr=matrix)

####################################
# MODULE-SPECIFIC HELPER FUNCTIONS #
####################################

##################
# FramesAnalyzer #
##################

def construct_batch_sequence(number_frames, batch_size):
    """
    Constructs a sequence of frame batches based on the total number of frames and batch size.

    Args:
        number_frames (int): Total number of frames.
        batch_size (int): Number of frames per batch.

    Returns:
        list: List of tuples representing frame batches.
    """
    number_batches, residual_frames = divmod(number_frames, batch_size)
    batches = [(batch_size*k-(batch_size-1), batch_size*k) for k in range(1, number_batches+1)]
    # add residual frames if any
    if residual_frames > 0:
        last_frame = batches[-1][1]
        residual_batch = (last_frame, last_frame + residual_frames)
        batches.append(residual_batch)

#####################
# AnalysesProcessor #
#####################

def _split_entry(res_atm: str) -> tuple:
    """
    Splits a residue-atom string into residue name and index.

    Args:
        res_atm (str): Input string in the format "RES_XX@YY".

    Returns:
        tuple: Residue name and index.
    
    Example:
        TRP_91@C -> [TRP, 91@C] -> (TRP, 91)
    """
    split_entry = res_atm.split("_")
    residue = split_entry[0]
    residue_index = split_entry[1].split("@")[0]
    return residue, residue_index

def _parse_line(line: str) -> str:
    """
    Parses a line of residue interaction data into a CSV-compatible format.

    Args:
        line (str): Line of interaction data.

    Returns:
        str: Parsed CSV-compatible line.
    """
    try:
        res_atm_A, _, _, res_atm_B, _, _, _, energy = line.split()
    except ValueError: # needed to add this line because cpptraj sometimes adds information on van der Waals (I think this is a bug)
        res_atm_A, _, _, res_atm_B, _, _, _, _, _, energy = line.split()

    res_A, res_A_index = _split_entry(res_atm_A)
    res_B, res_B_index = _split_entry(res_atm_B)
    return f"{res_A},{res_A_index},{res_B},{res_B_index},{energy}\n"

def read_lines(file_path: str) -> list:
    """
    Reads lines from a file, skipping the first line.

    Args:
        file_path (str): Path to the input file.

    Returns:
        list: List of lines from the file.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()[1::]
        return lines

def write_csv(lines: list, output_file_path: str) -> None:
    """
    Writes parsed interaction data to a CSV file.

    Args:
        lines (list): List of lines to write.
        output_file_path (str): Path to the output CSV file.
    """
    header = "residue_i,residue_i_index,residue_j,residue_j_index,energy\n"
    with open(output_file_path, "w") as output_file:
        output_file.write(header)
        for line in lines:
            parsed_line = _parse_line(line)
            output_file.write(parsed_line)

################################
# InteractionsToProbsConverter #
################################

def frames_from_name(file_name):
    """
    Extracts the start and end frame numbers from a file name.

    Args:
        file_name (str): File name containing frame information.

    Returns:
        tuple: Start and end frame numbers.
    """
    matched = search(r"(\d+)-(\d+)", file_name)
    start_frame = matched.group(1)
    end_frame = matched.group(2)
    return int(start_frame), int(end_frame)

###########
# Protein #
###########

def import_network_components(directory_path: str, config: dict):
    """
    Imports network components from the specified directory.

    Args:
        directory_path (str): Path to the directory containing network components.
        config (dict): Configuration dictionary specifying file and directory names.

    Returns:
        tuple: A tuple containing residues (tuple), interaction matrices (tuple), and probability matrices (tuple).

    Raises:
        FileNotFoundError: If the directory or expected files are not found.
        ValueError: If residue mapping or matrix loading fails due to invalid content.
    """
    # check if the directory exists
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The specified directory does not exist: {directory_path}")

    residues: tuple = None
    interaction_matrices = []
    probability_matrices = []

    # get all entries in the directory
    try:
        stored_items = [os.path.join(directory_path, dir) for dir in os.listdir(directory_path)]
    except OSError as e:
        raise FileNotFoundError(f"Could not list contents of directory {directory_path}: {e}")

    for entry in stored_items:
        if os.path.isdir(entry):
            # check for subdirectory contents
            try:
                matrices = os.listdir(entry)
            except OSError as e:
                raise FileNotFoundError(f"Could not access subdirectory {entry}: {e}")

            for matrix in matrices:
                matrix_path = os.path.join(entry, matrix)

                if config["ToMatricesConverter"]["interactions_matrix_name"] in matrix:
                    try:
                        interaction_matrices.append(np.load(matrix_path))
                    except Exception as e:
                        raise ValueError(f"Error loading interaction matrix from {matrix_path}: {e}")

                elif config["ToMatricesConverter"]["probabilities_matrix_name"] in matrix:
                    try:
                        probability_matrices.append(np.load(matrix_path))
                    except Exception as e:
                        raise ValueError(f"Error loading probability matrix from {matrix_path}: {e}")

        elif config["ToMatricesConverter"]["id_to_res_map_name"] in entry:
            # load residues mapping
            try:
                with open(entry, "r") as f:
                    residues = ast.literal_eval(f.readline())
            except FileNotFoundError:
                raise FileNotFoundError(f"Residue mapping file not found: {entry}")
            except SyntaxError as e:
                raise ValueError(f"Error parsing residue mapping in {entry}: {e}")

    # validate results
    if residues is None:
        raise ValueError("Residue mapping file is missing or empty.")
    if not interaction_matrices:
        raise ValueError("No interaction matrices were found in the specified directory.")
    if not probability_matrices:
        raise ValueError("No probability matrices were found in the specified directory.")

    return residues, tuple(interaction_matrices), tuple(probability_matrices)


def main():
    pass


if __name__ == "__main__":
    main()
