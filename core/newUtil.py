#!AllostericPathwayAnalyzer/venv/bin/python3
import os
from datetime import datetime
import numpy as np
from json import load
from concurrent.futures import as_completed

'''
PLEASE NOTE: This utilities file is a greatly simplified version of the util used for my research project.
The functions used here requrie optimisation, refactoring, and error handling.
'''

#############################
# MATRIX RELATED OPERATIONS #
#############################

# TEMPORARY FUNCTION CREATED FOR THE HPC PROJECT (WILL USE ONLY SOFTMAX FOR VECTOR NORMALISATION)
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

def process_elementwise(in_parallel=False, Executor=None, max_workers=None):

    if Executor is None:
        raise ValueError("An 'Executor' argument must be provided.")

    def inner(iterable, function, *extra_args, **extra_kwargs):

        nonlocal in_parallel
        nonlocal Executor

        results = []
        if in_parallel:
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
        
        return tuple(residues), tuple(interaction_matrices), tuple(probability_matrices)

def read_residues_file(f):
    with open(f, "r") as file:
        residues = tuple(map(str.strip,file.readline().split(",")))
    return residues


def main():
    pass

if __name__ == "__main__":
    main()
