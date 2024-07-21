import numpy as np
import logging

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
        return vector  # return the vector as-is if the sum is zero to avoid division by zero
    normalized_vector = vector / total
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

def extract_frames_range(file_name):
    from_ = file_name.index("(") + 1
    to_ = file_name.index(")")
    frames = file_name[from_:to_]
    start_frame, end_frame = frames.split("-")
    return int(start_frame), int(end_frame)

def main():
    pass

if __name__ == "__main__":
    main()
