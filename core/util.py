#!AllostericPathwayAnalyzer/venv/bin/python3

import atexit
import logging
from logging.config import dictConfig
import os
from datetime import datetime
import numpy as np
from json import load
from re import search
from concurrent.futures import as_completed
from copy import deepcopy
from numba import jit

#############################
# MATRIX RELATED OPERATIONS #
#############################

# TEMPORARY (will redo later)
def _softmax(matrix: np.array, axis=1):
    magnitudes_matrix = np.abs(matrix)
    shift_magnitudes_matrix = magnitudes_matrix - np.max(magnitudes_matrix, axis=axis, keepdims=True)
    exponents = np.exp(shift_magnitudes_matrix)
    probabilities_matrix = exponents / np.sum(exponents, axis=axis, keepdims=True)
    # Ensure the probability of going from residue i to itself is 0.0
    np.fill_diagonal(probabilities_matrix, 0.0)
    renormalized_matrix = normalize_vector(probabilities_matrix)
    return renormalized_matrix

# TODO: DELETE THIS FUNCTION, KEEP ONLY SOFTMAX FOR NORMALISATION
def normalize_vector(vector: np.array):
    if len(vector.shape) > 1:
        raise ValueError("Expected one-dimensional np.array")
    total = np.sum(vector)
    if total == 0:
        return np.zeros_like(vector)  # return a zero vector if the sum is zero to avoid division by zero
    normalized_vector = vector / total
    # Ensure the sum of the normalized vector is exactly 1
    if not np.isclose(np.sum(normalized_vector), 1.0):
        normalized_vector /= np.sum(normalized_vector)
    return normalized_vector

# def normalize_vector(vector: np.array):
#     return _softmax(vector)

def probabilities_from_interactions(matrix: np.array):
    return _softmax(matrix)


############################
# GENERAL HELPER FUNCTIONS #
############################

class CopyingTuple:
    def __init__(self, *args):
        self._data = tuple(args)

    def __getitem__(self, index):
        item = self._data[index]
        return deepcopy(item)
    
    def __contains__(self, element):
        return element in self._data
    
    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"CopyingTuple{self._data}"

def load_json_config(config_location: str) -> dict:
    with open(config_location, "r") as config_file:
        config = load(config_file)
    return config

# TODO: make more universal so that non-queue handlers can also be set up
def set_up_logging(config_location, logger_name):
    config = load_json_config(config_location)
    dictConfig(config)
    which_queue_handler = "queue_handler_construction_module" if logger_name == "network_construction_module" else "queue_handler_model"
    queue_handler = logging.getHandlerByName(which_queue_handler)
    if queue_handler is not None:
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)
        return logging.getLogger(logger_name)

def create_output_dir(output_directory_location: str, output_directory_name: str) -> str:
    # get the current time to ensure that the name of the output is unique
    current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

    # create the complete path to the output directory
    output_directory_path = os.path.join(output_directory_location, f"{output_directory_name}_{current_time}")

    # create the output directory
    os.makedirs(output_directory_path, exist_ok=True)

    return output_directory_path

def process_elementwise(in_parallel=False, Executor=None, max_workers = None):

    if Executor is None:
        raise ValueError("An 'Executor' argument must be provided.")

    def inner(iterable, function, *extra_args, **extra_kwargs):

        nonlocal in_parallel
        nonlocal Executor

        results = []
        if in_parallel:
            with Executor(max_workers=max_workers) as executor:
                tasks = [executor.submit(function, element, *extra_args, **extra_kwargs) for element in iterable]

                for future in as_completed(tasks):
                    result = future.result()
                    results.append(result)
        else:
            results = [function(element, *extra_args, **extra_kwargs) for element in iterable]
        return results
    
    return inner

##################################
# FILE-SPECIFIC HELPER FUNCTIONS #
##################################

def import_network_components(directory_path: str):

        residues = None
        interaction_matrices = []
        probability_matrices = []

        # Loop through files in the directory
        for filename in os.listdir(directory_path):
            filepath = os.path.join(directory_path, filename)
            
            if filename == "__pycache__" or not os.path.isdir(filepath) and not filename.endswith(".dat") and not filename.endswith(".npy"):
                continue

            # Load the residues variable from the Python file
            if filename.endswith(".dat"):
                residues = read_residues_file(filepath)

            # Loop through .npy files in subdirectories; load interaction and probability matrices
            elif os.path.isdir(filepath):
                # Loop through .npy files in subdirectories
                for npy_file in os.listdir(filepath):
                    path_to_npy_file = os.path.join(filepath, npy_file)
                    if npy_file.endswith(".npy"):
                        matrix = np.load(path_to_npy_file, allow_pickle=True)
                        if "interactions" in npy_file:
                            interaction_matrices.append(matrix)
                        if "probabilities" in npy_file:
                            probability_matrices.append(matrix)
        
        return CopyingTuple(*residues), CopyingTuple(*interaction_matrices), CopyingTuple(*probability_matrices)

def construct_batch_sequence(number_frames, batch_size):
    number_batches, residual_frames = divmod(number_frames, batch_size)
    batches = [(batch_size*k-(batch_size-1), batch_size*k) for k in range(1, number_batches+1)]
    # add residual frames if any
    if residual_frames > 0:
        last_frame = batches[-1][1]
        residual_batch = (last_frame, last_frame + residual_frames)
        batches.append(residual_batch)

def extract_frames_range(file_name):
    pattern = r"(\d+)-(\d+)"
    matched = search(pattern, file_name)
    start_frame = matched.group(1)
    end_frame = matched.group(2)
    return int(start_frame), int(end_frame)

# OPTIMIZE
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

def read_residues_file(f):
    with open(f, "r") as file:
        residues = tuple(map(str.strip,file.readline().split(",")))
    return residues


def main():
    pass

if __name__ == "__main__":
    main()
