import numpy as np

def softmax(matrix: np.array, axis=1):
    """
    Compute the softmax of each row of the input matrix.

    Parameters:
    matrix (np.array): Input matrix.
    axis (int): Axis along which to compute the softmax. Default is 1.

    Returns:
    np.array: The softmax of the input matrix.
    """
    magntitudes_matrix = np.abs(matrix)
    shift_magntitudes_matrix = magntitudes_matrix - np.max(magntitudes_matrix, axis=axis, keepdims=True)
    exponents = np.exp(shift_magntitudes_matrix)
    probabilities_matrix = exponents / np.sum(exponents, axis=axis, keepdims=True)
    # the following is done to ensure that the probability of going from residue i to itself is 0.0
    null_diag_probabilities_matrix = np.fill_diagonal(probabilities_matrix, 0.0)
    renormalized_matrix = normalize_row_vectors(null_diag_probabilities_matrix)
    return renormalized_matrix

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
    normalized_vector = vector / np.sum(vector)
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

class OneIndexedNpArray:
    """
    A wrapper for numpy arrays to use 1-based indexing.

    Attributes:
    array (np.array): The underlying 0-indexed numpy array.
    """

    def __init__(self, array):
        """
        Initialize the OneIndexedNpArray with a given array.

        Parameters:
        array (array-like): Input array.
        """
        self.array = np.asarray(array)
    
    def __getitem__(self, index):
        """
        Get item using 1-based index.

        Parameters:
        index (int or tuple of int): 1-based index or indices.

        Returns:
        The element of the array at the given 1-based index.
        """
        if isinstance(index, tuple):
            return self.array[tuple(i-1 for i in index)]
        return self.array[index-1]
    
    def __setitem__(self, index, value):
        """
        Set item using 1-based index.

        Parameters:
        index (int or tuple of int): 1-based index or indices.
        value: Value to set at the given index.
        """
        if isinstance(index, tuple):
            self.array[tuple(i-1 for i in index)] = value
        else:
            self.array[index-1] = value
    
    def __repr__(self):
        """
        Return the string representation of the array.

        Returns:
        str: String representation of the array.
        """
        return repr(self.array)

def extract_frames_range(file_name):
    from_ = file_name.index("(") + 1
    to_ = file_name.index(")")
    frames = file_name[from_:to_]
    start_frame, end_frame = frames.split("-")
    return int(start_frame), int(end_frame)

def default_match(file_name1: str, file_name2: str) -> bool:
    frames_range1 = extract_frames_range(file_name1)
    frames_range2 = extract_frames_range(file_name2)
    return frames_range1 == frames_range2
    
def main():
    pass


if __name__ == "__main__":
    main()