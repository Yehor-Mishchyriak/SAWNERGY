#!AllostericPathwayAnalyzer/venv/bin/python3

import numpy as np
import logging
from re import search
from concurrent.futures import as_completed

#############################
# MATRIX RELATED OPERATIONS #
#############################

def softmax(matrix: np.array, axis=1):
    """
    Compute the softmax of each row of the input matrix.

    Parameters:
    matrix (np.array): Input matrix.
    axis (int): Axis along which to compute the softmax. Default is 1.

    Returns:
    np.array: The softmax of the input matrix.
    """
    try:
        magnitudes_matrix = np.abs(matrix)
        shift_magnitudes_matrix = magnitudes_matrix - np.max(magnitudes_matrix, axis=axis, keepdims=True)
        exponents = np.exp(shift_magnitudes_matrix)
        probabilities_matrix = exponents / np.sum(exponents, axis=axis, keepdims=True)
        # Ensure the probability of going from residue i to itself is 0.0
        np.fill_diagonal(probabilities_matrix, 0.0)
        renormalized_matrix = normalize_row_vectors(probabilities_matrix)
        return renormalized_matrix
    except Exception as e:
        logging.error(f"Error in softmax function: {e}")
        raise

def transition_probs_from_interactions(matrix: np.array):
    """
    Compute transition probabilities from interaction matrix using softmax.

    Parameters:
    matrix (np.array): Interaction matrix.

    Returns:
    np.array: Transition probabilities.
    """
    return softmax(matrix)

# TODO: DELETE THIS FUNCTION, KEEP ONLY SOFTMAX FOR NORMALISATION
def normalize_vector(vector: np.array):
    """
    Normalize a 1D numpy array.

    Parameters:
    vector (np.array): Input vector.

    Returns:
    np.array: Normalized vector.

    Raises:
    ValueError: If the input array is not one-dimensional.
    """
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

def normalize_row_vectors(vectors: np.array):
    """
    Normalize each row vector of a 2D numpy array.

    Parameters:
    vectors (np.array): Input 2D array with row vectors.

    Returns:
    np.array: Array with normalized row vectors.
    """
    return np.apply_along_axis(func1d=normalize_vector, axis=1, arr=vectors)


####################
# HELPER FUNCTIONS #
####################
# TODO: DOC STRING, ERROR HANGLING, LOGGING
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

def extract_frames_range(file_name):
    pattern = r"(\d+)-(\d+)"
    matched = search(pattern, file_name)
    start_frame = matched.group(1)
    end_frame = matched.group(2)
    return int(start_frame), int(end_frame)


def main():
    pass

if __name__ == "__main__":
    main()
