import numpy as np

def softmax(matrix: np.array, axis=1):
    magntitudes_matrix = np.abs(matrix)
    shift_magntitudes_matrix = magntitudes_matrix - np.max(magntitudes_matrix, axis=axis, keepdims=True)
    exponents = np.exp(shift_magntitudes_matrix)
    return exponents / np.sum(exponents, axis=axis, keepdims=True)

def transition_probs_from_interactions(matrix: np.array):
    return softmax(matrix)

def normalize_vector(vector: np.array):
    if len(vector.shape) > 1:
        raise ValueError("Expected one dimensional np.array")
    normalized_vector = vector / np.sum(vector)
    return normalized_vector

def normalize_row_vectors(vectors: np.array):
    normalized_vectors = np.apply_along_axis(func1d=normalize_vector, axis=1, arr=vectors)
    return normalized_vectors

class OneIndexedNpArray:

    def __init__(self, array):
        self.array = np.asarray(array)
    
    def __getitem__(self, index):
        if isinstance(index, tuple):
            return self.array[tuple(i-1 for i in index)]
        return self.array[index-1]
    
    def __setitem__(self, index, value):
        if isinstance(index, tuple):
            self.array[tuple(i-1 for i in index)] = value
        else:
            self.array[index-1] = value
    
    def __repr__(self):
        return repr(self.array)

matrix = np.array([[1,2,3,4],
                  [1,42,12,8],
                  [0, 1, 12, 32]])

print(normalize_row_vectors(matrix))
