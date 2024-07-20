import os
from pathlib import Path
from typing import Union
from math import log, exp
import numpy as np
from statistics import mean, stdev
import util
# DON'T FORGET TO USE 1-INDEXED ARRAYS!!!
class Protein:

    def __init__(self, residues: dict, matrices_directory_path: Union[str, Path], epsilon: float = 1e-10) -> None:

        self.residues = residues
        self.number_residues = len(self.residues)
        self.epsilon = epsilon
        # prob matrices: row indices are "from", column indices are "to"
        self.interactions_matrices, self.probabilities_matrices = self.load_matrices(matrices_directory_path)
        self.number_matrices = len(self.probabilities_matrices)
        # Ensuring that no transition probability is zero, preventing issues with taking logarithms

    def load_matrices(self, directory_path):
        interactions_matrices = dict()
        probabilities_matrices = dict()
        for i, dir in enumerate(os.listdir(directory_path)):
            for npy_file in os.listdir(dir):
                path_to_npy_file = os.path.join(directory_path, npy_file)
                matrix = np.load(path_to_npy_file)
                if "interactions" in npy_file:
                    self.interactions_matrices[i] = matrix
                if "probabilities" in npy_file:
                    matrix += self.epsilon
                    self.probabilities_matrices[i] = matrix
        
        return interactions_matrices, probabilities_matrices

    def get_transition_probabilities_from_residue(self, residue_index: int) -> np.array:
        if residue_index not in self.residues:
            raise ValueError(f"Start residue index {residue_index} is not valid.")
        return self.transition_probabilities_matrix[residue_index, :]

    def get_matrices_probabilities_dist(self, residue_i: int, residue_j: int) -> np.array:
        energies = list()
        indexed_energies = dict()
        for index, matrix in self.interactions_matrices.keys():
            energy = matrix[residue_i, residue_j]
            indexed_energies[index] = energy
            energies.append(energy)
        mean_value = mean(energies)
        std_dev = stdev(energies)

        # not going to work because the destribution is not normal
        # instead round all the values to a certain limit for example 1 or 2 decimals, then
        # compute the destribution empirically based on the frequencies of the values


                                                # max number_iterations is number of residues # target_residues set of indices (ints)
    def allosteric_signal_path_builder(self, start: int, number_iterations: int, target_residues: set):
        pathway = [start]
        probability = 0
        which_matrix = np.random.randint(1, self.number_matrices)
        current_residue = start
        current_matrix = self.probabilities_matrices[which_matrix]
        for _ in range(number_iterations):
            residues_probabilities_dist = self.get_transition_probabilities_from_residue(current_residue)
            matrices_probabilities_dist = self.get_matrices_probabilities_dist(current_residue, next_residue)
            next_residue = current_residue
            while next_residue in pathway:
                next_residue = np.random.choice(range(1, self.number_residues), p=residues_probabilities_dist)
            next_matrix = np.random.choice(range(1, self.number_matrices), p=matrices_probabilities_dist)
            pathway.append(next_residue)
            probability += log(residues_probabilities_dist[next_residue])
            current_residue = next_residue
            current_matrix = next_matrix







def main():
    pass


if __name__ == "__main__":
    main()
