#!AllostericPathwayAnalyzer/venv/bin/python3

import atexit
import logging
from logging.config import dictConfig
import importlib.util
import os
import datetime
import numpy as np
from json import load
from re import search
from concurrent.futures import as_completed
from collections import Mapping
from copy import deepcopy


#############################
# MATRIX RELATED OPERATIONS #
#############################

def _softmax(matrix: np.array, axis=1):
    pass

def normalize_vector(vector: np.array):
    return _softmax(vector)

def probabilities_from_interactions(matrix: np.array):
    return _softmax(matrix)


############################
# GENERAL HELPER FUNCTIONS #
############################

def load_json_config(config_location: str) -> dict:
    with open(config_location, "r") as config_file:
        config = load(config_file)
    return config

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

def process_elementwise(in_parallel=False, Executor=None):

    if Executor is None:
        raise ValueError("An 'Executor' argument must be provided.")

    def inner(iterable, function, *extra_args, **extra_kwargs):

        nonlocal in_parallel
        nonlocal Executor

        results = []
        if in_parallel:
            with Executor() as executor:
                tasks = [executor.submit(function, element, *extra_args, **extra_kwargs) for element in iterable]

                for future in as_completed(tasks):
                    result = future.result()
                    results.append(result)
        else:
            results = [function(element, *extra_args, **extra_kwargs) for element in iterable]
        return results
    
    return inner

class FrozenDict(Mapping):
    def __init__(self, data):
        # Store a deep copy of the input dictionary to ensure immutability
        self._data = deepcopy(data)
        
    def __getitem__(self, key):
        # Return a deep copy of the value to prevent mutation
        return deepcopy(self._data[key])
    
    def __iter__(self):
        return iter(self._data)
    
    def __len__(self):
        return len(self._data)
    
    def __hash__(self):
        # FrozenDict is hashable based on the items in it, converted to a frozenset
        return hash(frozenset(self._data.items()))
    
    def __repr__(self):
        return f"FrozenDict({self._data})"

##################################
# FILE-SPECIFIC HELPER FUNCTIONS #
##################################

def import_network_directory_components(directory_path: str):
        residues = {}
        interaction_matrices = {}
        probability_matrices = {}
        
        interaction_matrix_index = 0
        probability_matrix_index = 0

        # Loop through files in the directory
        for filename in os.listdir(directory_path):
            filepath = os.path.join(directory_path, filename)
            
            if filename == "__pycache__" or not os.path.isdir(filepath) and not filename.endswith(".py") and not filename.endswith(".npy"):
                continue

            # Load the residues variable from the Python file
            if filename.endswith(".py"):
                spec = importlib.util.spec_from_file_location(filename[:-3], filepath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, 'residues'):
                    residues = module.residues

            # Loop through .npy files in subdirectories; load interaction and probability matrices
            elif os.path.isdir(filepath):
                # Loop through .npy files in subdirectories
                for npy_file in os.listdir(filepath):
                    path_to_npy_file = os.path.join(filepath, npy_file)
                    if npy_file.endswith(".npy"):
                        matrix = np.load(path_to_npy_file, allow_pickle=True)
                        if "interactions" in npy_file:
                            interaction_matrices[interaction_matrix_index] = matrix
                            interaction_matrix_index += 1
                        if "probabilities" in npy_file:
                            probability_matrices[probability_matrix_index]
                            probability_matrix_index += 1
        
        return FrozenDict(residues), FrozenDict(interaction_matrices), FrozenDict(probability_matrices)

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

def extract_residues_from_pdb(pdb_file: str, save_output: bool = False, output_directory: str = None):
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
                continue

    if save_output:
        if output_directory is None:
            output_directory = os.getcwd()
        output_file_path = os.path.join(output_directory, f"{os.path.splitext(os.path.basename(pdb_file))[0]}_residues.py")
        with open(output_file_path, 'w') as output_file:
            output_file.write(f"residues = {residues}")

    return residues

def main():
    pass

if __name__ == "__main__":
    main()
