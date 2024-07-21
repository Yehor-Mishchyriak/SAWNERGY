import os
import util
import numpy as np
from typing import Dict, Union, Set, Tuple
from math import log, exp
from collections import Counter


class Protein:

    def __init__(self, residues: Dict[int, str], matrices_directory_path: str, interactions_precision_limit: int = 1) -> None:
        # Residues is the map from residue indices to their respective names
        self.residues = residues
        self.number_residues = len(self.residues)
        self.interactions_matrices, self.probabilities_matrices = self._load_matrices(matrices_directory_path)
        self.number_matrices = len(self.probabilities_matrices)

        self.interactions_precision_limit = interactions_precision_limit
        # Note, interactions_precision_limit is required for empirically constructing a probability distribution for interactions matrices;
        # the more frequent certain energy values are across all the matrices, the higher the chance the associated matrices will be chosen
        # as intermediate matrices during the process of building the allosteric pathway

    def _load_matrices(self, directory_path):
        interactions_matrices = dict()
        probabilities_matrices = dict()

        for i, dir in enumerate(os.listdir(directory_path)):
            dir_path = os.path.join(directory_path, dir)
            for npy_file in os.listdir(dir_path):
                path_to_npy_file = os.path.join(dir_path, npy_file)
                matrix = np.load(path_to_npy_file)
                if "interactions" in npy_file:
                    interactions_matrices[i] = matrix
                if "probabilities" in npy_file:
                    probabilities_matrices[i] = matrix

        return interactions_matrices, probabilities_matrices

    def _get_transitions_prob_dist(self, residue_index: int, transition_probabilities_matrix) -> np.array:
        if residue_index not in self.residues:
            raise ValueError(f"Start residue index {residue_index} is not valid.")
        return transition_probabilities_matrix[residue_index, :]

    def _get_next_probability_matrix_and_selection_probability(self, preceding_residue: Union[None, int], current_residue: int, next_residue: int, current_matrix_index: int) -> Tuple[np.array, float]:
        indexed_rounded_energies_btw_current_next = dict()
        rounded_energy_counts_btw_current_next = Counter()

        while preceding_residue is None or preceding_residue == current_residue or preceding_residue == next_residue:
            preceding_residue = np.random.randint(0, self.number_residues)

        latest_energy_btw_preceding_current = self.interactions_matrices[current_matrix_index][preceding_residue, current_residue]

        for which_matrix, matrix in self.interactions_matrices.items():
            rounded_matrix = np.round(matrix, decimals=self.interactions_precision_limit)
            rounded_energy_btw_current_next = rounded_matrix[current_residue, next_residue]
            indexed_rounded_energies_btw_current_next[which_matrix] = rounded_energy_btw_current_next
            rounded_energy_counts_btw_current_next[rounded_energy_btw_current_next] += 1

        rounded_energies_probability_dist = {rounded_energy: count / self.number_matrices for rounded_energy, count in rounded_energy_counts_btw_current_next.items()}
        rounded_energies = np.array(list(rounded_energies_probability_dist.keys()))
        probabilities = np.array(list(rounded_energies_probability_dist.values()))

        drawn_energy = np.random.choice(rounded_energies, p=probabilities)

        filtered_matching_matrices = [
            which_matrix for which_matrix, energy in indexed_rounded_energies_btw_current_next.items() if energy == drawn_energy
        ]

        current_minimal_difference = float("inf")
        selected_matrix_index = None

        for which_matrix in filtered_matching_matrices:
            observed_energy = self.interactions_matrices[which_matrix][preceding_residue, current_residue]
            if abs(observed_energy - latest_energy_btw_preceding_current) < current_minimal_difference:
                current_minimal_difference = abs(observed_energy - latest_energy_btw_preceding_current)
                selected_matrix_index = which_matrix

        matrix_selection_probability = 1 / len(filtered_matching_matrices)
        selected_probability_matrix = self.probabilities_matrices[selected_matrix_index]

        return selected_probability_matrix, matrix_selection_probability

    def generate_allosteric_signal_pathway(self, start: int, number_iterations: Union[None, int] = None, target_residues: Union[None, Set[int]] = None) -> Tuple[str, float]:
        if number_iterations is None:
            number_iterations = self.number_residues

        if target_residues is None:
            target_residues = set()

        pathway = [start]
        log_aggregated_probability = 0

        current_matrix_index = np.random.randint(0, self.number_matrices)
        current_matrix = self.probabilities_matrices[current_matrix_index]  # Select randomly as the start (we don't know at what moment the binding occurred)
        preceding_residue = None
        current_residue = start
        next_residue = None

        for _ in range(number_iterations):
            residues_probability_vector = self._get_transitions_prob_dist(current_residue, current_matrix)
            residues_probability_vector[pathway] = 0.0  # Avoid loops by setting already visited residues to 0

            probability_vector_given_current_pathway = util.normalize_vector(residues_probability_vector)

            next_residue = np.random.choice(range(0, self.number_residues), p=probability_vector_given_current_pathway)
            residue_selection_probability = probability_vector_given_current_pathway[next_residue]

            next_matrix, matrix_selection_probability = self._get_next_probability_matrix_and_selection_probability(preceding_residue, current_residue, next_residue, current_matrix_index)

            pathway.append(next_residue)  # Extend the path

            log_aggregated_probability += log(matrix_selection_probability) + log(residue_selection_probability)

            if next_residue in target_residues:
                break

            current_matrix = next_matrix
            preceding_residue = current_residue
            current_residue = next_residue
            next_residue = None

        final_pathway = self.format_pathway(pathway)
        aggregated_probability = exp(log_aggregated_probability)

        return final_pathway, aggregated_probability

    def format_pathway(self, pathway: list) -> str:
        path = ""
        for residue_index in pathway:
            path += f"({residue_index+1}-{self.residues[residue_index]})"
        
        return path


def main():
    pass


if __name__ == "__main__":
    main()
