#!AllostericPathwayAnalyzer/venv/bin/python3

import os
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Union, Set, Tuple
from math import log, exp
from collections import Counter
import util

# Set up logging
logging.basicConfig(
    filename='protein.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class Protein:
    """
    A class to represent a protein and analyze its allosteric pathways.

    Attributes:
        residues (Dict[int, str]): A dictionary mapping residue indices to their respective names.
        number_residues (int): The total number of residues in the protein.
        interactions_matrices (Dict[int, np.array]): Interaction matrices loaded from files.
        probabilities_matrices (Dict[int, np.array]): Probability matrices loaded from files.
        number_matrices (int): The number of matrices loaded.
        interactions_precision_limit (int): Precision limit for rounding interaction energies.
    """

    def __init__(self, residues: Dict[int, str], matrices_directory_path: str, interactions_precision_limit: int = 1, random_seed: Union[int, None] = None) -> None:
        """
        Initialize the Protein instance with residues and matrices.

        Args:
            residues (Dict[int, str]): A dictionary of residue indices and their names.
            matrices_directory_path (str): Path to the directory containing matrices.
            interactions_precision_limit (int, optional): Precision limit for interactions. Defaults to 1.
        """
        try:
            self.residues = residues
            self.number_residues = len(self.residues)
            self.interactions_matrices, self.probabilities_matrices = self._load_matrices(matrices_directory_path)
            self.number_matrices = len(self.probabilities_matrices)
            self.interactions_precision_limit = interactions_precision_limit
            self.random_seed = Protein.set_random_seed(random_seed)
        except Exception as e:
            logging.error(f"Error initializing Protein: {e}")
            raise
    
    @staticmethod
    def set_random_seed(seed: Union[int, None] = None) -> None:
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): The seed value to set.
        """
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        np.random.seed(seed)
        logging.info(f"Random seed set to: {seed}")

    def _load_residues_list():
        # take this function from the FramesAnalyzer.py
        pass

    def _load_matrices(self, directory_path: str) -> Tuple[Dict[int, np.array], Dict[int, np.array]]:
        """
        Load interaction and probability matrices from the given directory.

        Args:
            directory_path (str): Path to the directory containing matrices.

        Returns:
            Tuple[Dict[int, np.array], Dict[int, np.array]]: A tuple containing dictionaries of interaction and probability matrices.

        Raises:
            OSError: If there's an error accessing the directory or files.
        """
        try:
            interactions_matrices = {}
            probabilities_matrices = {}
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
        except OSError as e:
            logging.error(f"Error loading matrices from {directory_path}: {e}")
            raise

    def _get_transitions_prob_dist(self, residue_index: int, transition_probabilities_matrix: np.array) -> np.array:
        """
        Get the transition probability distribution for a given residue index.

        Args:
            residue_index (int): The index of the residue.
            transition_probabilities_matrix (np.array): The matrix of transition probabilities.

        Returns:
            np.array: The transition probability vector for the given residue index.

        Raises:
            ValueError: If the residue index is invalid.
        """
        if residue_index not in self.residues:
            raise ValueError(f"Start residue index {residue_index} is not valid.")
        return transition_probabilities_matrix[residue_index, :]

    def _get_next_probability_matrix_and_selection_probability(
            self, preceding_residue: Union[None, int], current_residue: int, next_residue: int, current_matrix_index: int
    ) -> Tuple[np.array, float]:
        """
        Get the next probability matrix and the selection probability.

        Args:
            preceding_residue (Union[None, int]): The preceding residue index.
            current_residue (int): The current residue index.
            next_residue (int): The next residue index.
            current_matrix_index (int): The index of the current matrix.

        Returns:
            Tuple[np.array, float]: The next probability matrix and the selection probability.
        """
        indexed_rounded_energies_btw_current_next = {}
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

    def generate_allosteric_signal_pathway(
            self, start: int, number_iterations: Union[None, int] = None, target_residues: Union[None, Set[int]] = None
    ) -> Tuple[str, float]:
        """
        Generate the allosteric signal pathway starting from a residue.

        Args:
            start (int): The starting residue index.
            number_iterations (Union[None, int], optional): Number of iterations. Defaults to the number of residues.
            target_residues (Union[None, Set[int]], optional): Set of target residues. Defaults to an empty set.

        Returns:
            Tuple[str, float]: The pathway as a formatted string and the aggregated probability.
        """
        try:
            if number_iterations is None:
                number_iterations = self.number_residues

            if target_residues is None:
                target_residues = set()

            pathway = [start]
            log_aggregated_probability = 0

            current_matrix_index = np.random.randint(0, self.number_matrices)
            current_matrix = self.probabilities_matrices[current_matrix_index]
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

                pathway.append(next_residue)

                log_aggregated_probability += log(matrix_selection_probability) + log(residue_selection_probability)

                if next_residue in target_residues:
                    break

                current_matrix = next_matrix
                preceding_residue = current_residue
                current_residue = next_residue
                next_residue = None

            final_pathway = self._format_pathway(pathway)
            aggregated_probability = exp(log_aggregated_probability)

            return final_pathway, aggregated_probability
        except Exception as e:
            logging.error(f"Error generating allosteric signal pathway: {e}")
            raise

    def _format_pathway(self, pathway: list) -> str:
        """
        Format the pathway for display.

        Args:
            pathway (list): List of residue indices in the pathway.

        Returns:
            str: The formatted pathway as a string.
        """
        path = ""
        for residue_index in pathway:
            path += f"({residue_index + 1}-{self.residues[residue_index]})"
        return path

    def create_pathways(self, perturbed_residue: int, number_iterations: Union[None, int] = None, 
                        target_residues: Union[None, Set[int]] = None, number_pathways: int = 100, 
                        filter_out_improbable: bool = True, percentage_kept: float = 0.1, 
                        output_directory: Union[None, str] = None) -> str:
        """
        Create allosteric signal pathways starting from a perturbed residue.

        Args:
            perturbed_residue (int): The starting perturbed residue index.
            number_iterations (Union[None, int], optional): Number of iterations for each pathway generation. Defaults to None.
            target_residues (Union[None, Set[int]], optional): Set of target residues. Defaults to None.
            number_pathways (int, optional): Number of pathways to generate. Defaults to 100.
            filter_out_improbable (bool, optional): Whether to filter out improbable pathways. Defaults to True.
            percentage_kept (float, optional): The percentage of most probable pathways to keep. Defaults to 0.1.
            output_directory (Union[None, str], optional): Directory to save the generated pathways. Defaults to None.

        Returns:
            str: The path to the output directory where the pathways file is saved.
        
        Raises:
            ValueError: If any of the provided parameters are invalid.
        """
        try:
            # Ensure percentage_kept is within valid range
            if not (0 < percentage_kept <= 1):
                raise ValueError("percentage_kept must be between 0 and 1 (exclusive).")

            logging.info("Starting pathway generation.")
            logging.info(f"Parameters: perturbed_residue={perturbed_residue}, number_iterations={number_iterations}, "
                        f"target_residues={target_residues}, number_pathways={number_pathways}, "
                        f"filter_out_improbable={filter_out_improbable}, percentage_kept={percentage_kept}, "
                        f"output_directory={output_directory}")

            generated_pathways = []
            for i in range(number_pathways):
                pathway = self.generate_allosteric_signal_pathway(perturbed_residue, number_iterations, target_residues)
                generated_pathways.append(pathway)
                logging.debug(f"Generated pathway {i+1}/{number_pathways}: {pathway}")

            # Sort pathways by probability in descending order
            generated_pathways.sort(key=lambda x: x[1], reverse=True)

            # Filter out improbable pathways if required
            if filter_out_improbable:
                number_kept = int(number_pathways * percentage_kept)
                most_probable_pathways = generated_pathways[:number_kept]
                logging.info(f"Filtered out improbable pathways. Keeping {number_kept} pathways.")
            else:
                most_probable_pathways = generated_pathways

            current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
            output_file_name = f"from_({perturbed_residue}-{self.residues[perturbed_residue]})_{number_pathways}_pathways_{current_time}.txt"
            if output_directory is None:
                output_directory = os.getcwd()
            save_output_to = os.path.join(output_directory, output_file_name)

            # Write pathways to output file
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
                    output.write(f"{index + 1}) {path}\n")
                    logging.debug(f"Pathway {index + 1}: {path} - Probability: {probability}")

            logging.info(f"Pathways saved to {save_output_to}")
            return output_directory

        except Exception as e:
            logging.error(f"Error creating pathways: {e}")
            raise


def main():
    # testing
    p53 = Protein()


if __name__ == "__main__":
    main()
