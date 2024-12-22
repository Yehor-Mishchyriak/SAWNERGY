#!AllostericPathwayAnalyzer/venv/bin/python3

# TODO: GO OVER THE ENTIRE FILE: OPTIMIZE EACH FUNCTION, CLEAN THE CODE, WRITE THOROUGH DOCSTRINGS

import os
import atexit
import logging
from logging.config import dictConfig
from datetime import datetime
import numpy as np
from json import load
from concurrent.futures import as_completed
from re import search

"""
    STILL LEFT TO BE DOCUMENTED
"""
###################################################
# General function distributed across the modules #
###################################################
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

"""
    STILL LEFT TO BE OPTILIZED AND DOCUMENTED
"""
####################################
# MODULE-SPECIFIC HELPER FUNCTIONS #
####################################

def construct_batch_sequence(number_frames, batch_size):
    number_batches, residual_frames = divmod(number_frames, batch_size)
    batches = [(batch_size*k-(batch_size-1), batch_size*k) for k in range(1, number_batches+1)]
    # add residual frames if any
    if residual_frames > 0:
        last_frame = batches[-1][1]
        residual_batch = (last_frame, last_frame + residual_frames)
        batches.append(residual_batch)

def import_network_components(directory_path: str):
    residues = None
    interaction_matrices = []
    probability_matrices = []

    return residues, interaction_matrices, probability_matrices

def read_residues_file(f):
    with open(f, "r") as file:
        residues = tuple(map(str.strip,file.readline().split(",")))
    return residues

def extract_frames_range(file_name):
    pattern = r"(\d+)-(\d+)"
    matched = search(pattern, file_name)
    start_frame = matched.group(1)
    end_frame = matched.group(2)
    return int(start_frame), int(end_frame)

def extract_residues_from_pdb(pdb_file: str, file_path: str = None):
    residues = {}
    with open(pdb_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            try:
                _, _, _, residue, index, _, _, _, _, _ = line.split()
                residue = residue.strip()
                # note need to subtract 1 to keep it zero-indexed to match np.array matrices
                index = int(index.strip()) - 1
                residues[index] = residue
            except ValueError:  # in case the line being parsed is of a different format
                # add proper handling
                raise
    residues = tuple(sorted(list(residues.items()), key=lambda index_residue: index_residue[0]))
    with open(file_path, 'w') as output_file:
            output_file.write(f"residues = {residues}")

#####################
# AnalysesProcessor #
#####################

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

def _parse_line(line: str) -> str:
    try:
        res_atm_A, _, _, res_atm_B, _, _, _, energy = line.split()
    except ValueError: # needed to add this line because cpptraj sometimes adds information on van der Waals (I think this is a bug)
        res_atm_A, _, _, res_atm_B, _, _, _, _, _, energy = line.split()

    res_A, res_A_index = _split_residue(res_atm_A)
    res_B, res_B_index = _split_residue(res_atm_B)
    return f"{res_A},{res_A_index},{res_B},{res_B_index},{energy}\n"

def _split_residue(res_atm: str) -> tuple:
    residue = res_atm.split("_")[0]
    residue_index = res_atm.split("_")[1].split("@")[0]
    return residue, residue_index


def main():
    pass


if __name__ == "__main__":
    main()
