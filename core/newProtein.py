import os
import numpy as np
from datetime import datetime
from typing import Dict, Union, Set, Tuple
from math import log, exp
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import util

root_config = util.load_json_config("AllostericPathwayAnalyzer/configs/root.json")
logger = util.set_up_logging("AllostericPathwayAnalyzer/configs/logging.json", "network_construction_module")

class Protein:

    def __init__(self, network_directory_path: str, interactions_precision_limit_decimals: int = 1, seed: Union[int, None] = None) -> None:

        network_components = util.import_network_components(network_directory_path)

        self.residues: tuple = network_components[0]
        self.interactions_matrices: tuple = network_components[0]
        self.probabilities_matrices: tuple = network_components[0]

        self.number_residues: int = len(self.residues)
        self.residues_range: int = self.number_residues - 1

        self.number_matrices: int = len(self.probabilities_matrices)
        self.interactions_precision_limit_decimals: float = interactions_precision_limit_decimals

        self.seed: int = seed
        Protein._set_seed(seed)

    @staticmethod
    def _set_seed(seed: Union[int, None] = None) -> None:
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        np.random.seed(seed)

    def _format_pathway(self, pathway: list, VMD_interpretable: bool = True) -> str:
        vmd = lambda id: f" {id + 1}"
        default = lambda id: f" ({id + 1}-{self.residues[id]})"
        if VMD_interpretable:
            path = "resid"
            format_ = vmd
        else:
            path = ""
            format_ = default

        for residue_index in pathway:
            path += format_(residue_index)
        return path

    def _get_transitions_prob_dist(self, residue_index: int, transition_probabilities_matrix: np.array) -> np.array:
        if residue_index not in self.residues:
            raise ValueError(f"Start residue index {residue_index} is not valid.")
        return transition_probabilities_matrix[residue_index, :]

    def _get_next_probability_matrix_and_selection_probability(
            self, preceding_residue: Union[None, int], current_residue: int, next_residue: int, current_matrix_index: int):

        # Ensure preceding_residue is valid
        if preceding_residue is None or preceding_residue in (current_residue, next_residue):
            valid_residues = [i for i in range(self.residues_range) if i not in (current_residue, next_residue)]
            preceding_residue = np.random.choice(valid_residues)

        # Get the last observed energy between preceding and current residue
        last_observed_energy_btw_preceding_current = self.interactions_matrices[current_matrix_index][preceding_residue, current_residue]

        # Calculate rounded energies and their probabilities
        rounded_energy_counts_btw_current_next = np.round([matrix[current_residue, next_residue] 
                                                        for matrix in self.interactions_matrices],
                                                        decimals=self.interactions_precision_limit_decimals)
        
        # Get unique rounded energies and their counts
        unique, counts = np.unique(rounded_energy_counts_btw_current_next, return_counts=True)

        # Calculate probabilities as count / number of matrices
        probabilities = counts / self.number_matrices

        # Use `unique` as the list of rounded energies directly
        rounded_energies = unique

        # Draw an energy value based on the probability distribution
        drawn_energy = np.random.choice(rounded_energies, p=probabilities)

        # Select the matrix index where the energy matches
        selected_matrix_index, _ = min(
            [(which_matrix, abs(matrix[preceding_residue, current_residue] - last_observed_energy_btw_preceding_current))
            for which_matrix, matrix in enumerate(self.interactions_matrices)
            if np.round(matrix[current_residue, next_residue], decimals=self.interactions_precision_limit_decimals) == drawn_energy],
            key=lambda matrix_difference_pair: matrix_difference_pair[1])

        # Calculate matrix selection probability
        matrix_selection_probability = 1 / counts[np.where(rounded_energies == drawn_energy)][0]

        return selected_matrix_index, matrix_selection_probability

    def _generate_allosteric_signal_pathway(
            self, start: int, number_iterations: Union[None, int] = None, target_residues: Union[None, Set[int]] = None
    ) -> Tuple[str, float]:

        pathway = [start]
        log_aggregated_probability = 0

        current_matrix_index = np.random.randint(0, self.number_matrices)
        preceding_residue = None
        current_residue = start
        next_residue = None

        for _ in range(number_iterations):
            probability_vector = self._get_transitions_prob_dist(current_residue, self.probabilities_matrices[current_matrix_index])
            probability_vector[pathway] = 0.0  # Avoid loops by setting already visited residues to 0
            probability_vector = util.normalize_vector(probability_vector)

            next_residue = np.random.choice(range(0, self.residues_range), p=probability_vector)
            residue_selection_probability = probability_vector[next_residue]

            next_matrix_index, matrix_selection_probability = self._get_next_probability_matrix_and_selection_probability(preceding_residue, current_residue, next_residue, current_matrix_index)

            pathway.append(next_residue)
            log_aggregated_probability += log(residue_selection_probability) + log(matrix_selection_probability)

            if next_residue in target_residues:
                break
            
            current_matrix_index = next_matrix_index
            preceding_residue = current_residue
            current_residue = next_residue
            next_residue = None

        aggregated_probability = exp(log_aggregated_probability)

        return pathway, aggregated_probability

    # TODO: ensure all the user-input values are valid
    def create_pathways(self, perturbed_residue: int, number_iterations: Union[None, int] = None, 
                        target_residues: Union[None, Set[int]] = None, number_pathways: int = 100, 
                        filter_out_improbable: bool = True, percentage_kept: float = 0.1, 
                        output_directory: Union[None, str] = None) -> str:
        # account for zero-indexing
        if number_iterations is None:
            number_iterations = self.residues_range - 1

        generated_pathways = util.process_elementwise(in_parallel=True, Executor=ProcessPoolExecutor)(range(number_pathways), self._generate_allosteric_signal_pathway, perturbed_residue, number_iterations, target_residues)
        generated_pathways.sort(key=lambda x: x[1], reverse=True)

        if filter_out_improbable:
            number_kept = int(number_pathways * percentage_kept)
            most_probable_pathways = generated_pathways[:number_kept]
        else:
            most_probable_pathways = generated_pathways

        current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        output_file_name = f"from_({perturbed_residue}-{self.residues[perturbed_residue-1]})_{number_pathways}_pathways_{current_time}.txt"
        if output_directory is None:
            output_directory = os.getcwd()
        save_output_to = os.path.join(output_directory, output_file_name)

        with open(save_output_to, "w") as output:
            header = f"""Generated allosteric pathways sorted from more probable to less probable (top to bottom)
            The following parameters were used:
            perturbed_residue: {perturbed_residue}
            number_iterations: {number_iterations}
            target_residues: {target_residues}
            number_pathways: {number_pathways}
            filter_out_improbable: {filter_out_improbable}
            percentage_kept: {percentage_kept}
            output_directory: {output_directory}
            random_seed: {self.random_seed}
            """
            output.write(header + "\n")
            for index, result in enumerate(most_probable_pathways):
                path, probability = result
                output.write(f"{index + 1}. {path}\n")

        return output_directory


def main():
    pass


if __name__ == "__main__":
    main()
