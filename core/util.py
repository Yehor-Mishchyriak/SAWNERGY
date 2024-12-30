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

####################################################
# General functions distributed across the modules #
####################################################
def load_json_config(config_location: str) -> dict:
    with open(config_location, "r") as config_file:
        config = load(config_file)
    return config

def create_output_dir(output_directory_location: str, output_directory_name: str) -> str:
    # get the current time to ensure that the name of the output is unique
    current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

    # create the complete path to the output directory
    output_directory_path = os.path.join(output_directory_location, f"{output_directory_name}_{current_time}")

    # create the output directory
    os.makedirs(output_directory_path, exist_ok=True)

    return output_directory_path

def set_up_logging(config_location, logger_name, handler_name):
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

            print("Running in parallel")
            with Executor(max_workers=max_workers) as executor:
                tasks = [executor.submit(function, element, *extra_args, **extra_kwargs) for element in iterable]

                for future in as_completed(tasks):
                    result = future.result()
                    results.append(result)
        else:
            print("Running sequentially")
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
    # x is a vector
    max_x_i = np.max(x)
    # elementwise exponentiation for vector x
    exp_x = np.exp(x - max_x_i) # (note the trick with the uniform subtraction of the largest entry)
    return exp_x / np.sum(exp_x) # normalize each entry in the exponent space

# numerically stable log-softmax: avoiding overflow with large exponents
# as well as avoiding underflow when multiplying small probabilities
# note the necessity of exponentiating eventually for converting back from the log space
def _log_softmax(x: np.array):
    # x is a vector
    max_x_i = np.max(x)
    # add up the logs of sums
    log_sum_exp = max_x_i + np.log(np.sum(np.exp(x - max_x_i)))
    return x - log_sum_exp # normalize in the log space

def normalize_vector(v: np.array):
    return _softmax(v)

def normalize_rows(matrix: np.array):
    return np.apply_along_axis(normalize_vector, axis=1, arr=matrix)

####################################
# MODULE-SPECIFIC HELPER FUNCTIONS #
####################################

##################
# FramesAnalyzer #
##################

def construct_batch_sequence(number_frames, batch_size):
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
    Example:
    TRP_91@C -> [TRP, 91@C] -> (TRP, 91)
    """
    split_entry = res_atm.split("_")
    residue = split_entry[0]
    residue_index = split_entry[1].split("@")[0]
    return residue, residue_index

def _parse_line(line: str) -> str:
    try:
        res_atm_A, _, _, res_atm_B, _, _, _, energy = line.split()
    except ValueError: # needed to add this line because cpptraj sometimes adds information on van der Waals (I think this is a bug)
        res_atm_A, _, _, res_atm_B, _, _, _, _, _, energy = line.split()

    res_A, res_A_index = _split_entry(res_atm_A)
    res_B, res_B_index = _split_entry(res_atm_B)
    return f"{res_A},{res_A_index},{res_B},{res_B_index},{energy}\n"

def read_lines(file_path: str) -> list:
    with open(file_path, "r") as file:
        lines = file.readlines()[1::]
        return lines

def write_csv(lines: list, output_file_path: str) -> None:
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
    matched = search(r"(\d+)-(\d+)", file_name)
    start_frame = matched.group(1)
    end_frame = matched.group(2)
    return int(start_frame), int(end_frame)

###########
# Protein #
###########

def import_network_components(directory_path: str):
    raise NotImplemented


def main():
    pass


if __name__ == "__main__":
    main()
