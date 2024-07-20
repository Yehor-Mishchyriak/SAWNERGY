import os
import util
import numpy as np
from typing import Dict, Union, Set
from math import log, exp
from collections import Counter

# don't forget to fix indexing at the end, make it 1-indexed
class Protein:

    def __init__(self, residues: Dict[int, str], matrices_directory_path: str, interactions_precision_limit: int = 1) -> None:
        # residues is the map from residue indices to their respective names
        self.residues = residues
        self.number_residues = len(self.residues)
        self.interactions_matrices, self.probabilities_matrices = self._load_matrices(matrices_directory_path)
        self.number_matrices = len(self.probabilities_matrices)

        self.interactions_precision_limit = interactions_precision_limit
        # Note, interactions_precision_limit is required for empirically constructing a probability distribution for interactions matrices;
        # the more frequent certain energy values are accross all the matrices, the higher the chance the associated matrices will be chosen
        # as an intermediate matrices during the process of building the allosteric pathway

    def _load_matrices(self, directory_path):
        
        interactions_matrices = dict()
        probabilities_matrices = dict()
        
        for i, dir in enumerate(os.listdir(directory_path)):
            for npy_file in os.listdir(dir):
                path_to_npy_file = os.path.join(directory_path, npy_file)
                matrix = np.load(path_to_npy_file)
                if "interactions" in npy_file:
                    self.interactions_matrices[i] = matrix
                if "probabilities" in npy_file:
                    self.probabilities_matrices[i] = matrix
        
        return interactions_matrices, probabilities_matrices

    def _get_transitions_prob_dist(self, residue_index: int, transition_probabilities_matrix) -> np.array:
        if residue_index not in self.residues:
            raise ValueError(f"Start residue index {residue_index} is not valid.")
        return transition_probabilities_matrix[residue_index, :]

    def _get_next_probability_matrix(self, preceding_residue: Union[None, int], current_residue: int, next_residue: int, current_matrix_index: int) -> np.array:

        indexed_rounded_energies_btw_current_next = dict()
        rounded_energy_counts_btw_current_next = Counter()

        while preceding_residue == None or preceding_residue == current_residue or preceding_residue == next_residue:
            preceding_residue = np.random.randint(0, self.number_residues)
        
        latest_energy_btw_preceding_current = self.interactions_matrices[current_matrix_index][preceding_residue, current_residue]

        for which_matrix, matrix in self.interactions_matrices.keys():

            rounded_matrix = np.round(matrix, decimals=self.interactions_precision_limit)
            rounded_energy_btw_current_next = rounded_matrix[current_residue, next_residue]
            indexed_rounded_energies_btw_current_next[which_matrix] = rounded_energy_btw_current_next
            rounded_energy_counts_btw_current_next[rounded_energy_btw_current_next] += 1

        rounded_energies_probability_dist = {rounded_energy: count / self.number_matrices for rounded_energy, count in rounded_energy_counts_btw_current_next.items()}
        rounded_energies = np.array(list(rounded_energies_probability_dist.keys()))
        probabilities = np.array(list(rounded_energies_probability_dist.values()))

        drawn_energy = np.random.choice(rounded_energies, p=probabilities)

        filtered_matching_matrices = filter(lambda _, energy: energy == drawn_energy, indexed_rounded_energies_btw_current_next.items())

        current_minimal_difference = float("inf")
        corresponding_probability_matrix: np.array = None
        for which_matrix, _ in filtered_matching_matrices:
            observed_energy = self.interactions_matrices[which_matrix][preceding_residue, current_residue]
            if abs(observed_energy - latest_energy_btw_preceding_current) < current_minimal_difference:
                current_minimal_difference = observed_energy
                corresponding_probability_matrix = self.probabilities_matrices[which_matrix]
        
        return corresponding_probability_matrix
        
    def allosteric_signal_path_builder(self, start: int, number_iterations: Union[None, int] = None, target_residues: Union[None, Set[int]] = None):

        if number_iterations is None:
            number_iterations = self.number_residues
        
        if target_residues is None:
            target_residues = set()

        pathway = [start]
        aggregated_probability = 0

        current_matrix_index = np.random.randint(0, self.number_matrices)
        current_matrix = self.probabilities_matrices[current_matrix_index] # select randomly as the start (we don't know at what moment the bidning occurred)
        preceding_residue: int = None
        current_residue: int = start
        next_residue: int = None

        for _ in range(number_iterations):

            residues_probability_vector = self._get_transitions_prob_dist(current_residue, current_matrix)
            residues_probability_vector[pathway] = 0.0
            probability_vector_given_current_pathway = util.normalize_vector(residues_probability_vector)
            
            next_residue = np.random.choice(range(0, self.number_residues), p=probability_vector_given_current_pathway)
            next_matrix = self._get_next_probability_matrix(preceding_residue, current_residue, next_residue, current_matrix_index)

            pathway.append(next_residue) # extend the path

            aggregated_probability += "I don't know how to compute it taking into account both the probabilities for the next residues and probabilities for the next matrices"

            if next_residue in target_residues:
                break

            current_matrix = next_matrix
            preceding_residue = current_residue
            current_residue = next_residue
            next_residue = None

        one_based_indexed_pathway = list(map(lambda x: x+1, pathway))
        aggregated_probability = exp(aggregated_probability)
        
        return one_based_indexed_pathway, aggregated_probability


def main():
    pass


if __name__ == "__main__":
    main()
