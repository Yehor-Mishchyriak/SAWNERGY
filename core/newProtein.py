import os
import numpy as np
from typing import Union, Tuple, List
from math import log, exp
from concurrent.futures import ProcessPoolExecutor
import util
from itertools import chain

root_config = util.load_json_config("/home/yehor/research_project/AllostericPathwayAnalyzer/configs/root.json")
# logger = util.set_up_logging("AllostericPathwayAnalyzer/configs/logging.json", "network_construction_module")

class Protein:

    def __init__(self, network_directory_path: str, interactions_precision_limit_decimals: int = 1, seed: Union[int, None] = None) -> None:

        network_components: Tuple[util.CopyingTuple, util.CopyingTuple, util.CopyingTuple] = util.import_network_components(network_directory_path)
        self.residues = network_components[0]
        self.interactions_matrices = network_components[1]
        self.probabilities_matrices = network_components[2]

        self.number_residues: int = len(self.residues)
        # NOTE: FIGURE OUT IF IT HAS TO BE JUST self.number_residues or you need to decrement it by 1
        self.residues_range: int = self.number_residues - 1

        self.number_matrices: int = len(self.probabilities_matrices)
        self.interactions_precision_limit_decimals: int = interactions_precision_limit_decimals

        self.seed: int = self._set_seed(seed)
        self.output_file: str = os.path.join(util.create_output_dir(root_config["GLOBAL"]["output_directory_path"],
                                                                    root_config["Protein"]["output_directory_name"]),
                                                                    root_config["Protein"]["pathways_file_name"])

    def _set_seed(self, seed: Union[int, None] = None) -> int:
        seed = np.random.randint(0, 2**32 - 1) if seed is None else seed
        self.seed = seed
        np.random.seed(seed)
        return seed

    def _format_pathway(self, visited_residues_indices: list, VMD_compatible: bool = True) -> str:
        if VMD_compatible:
            path = "resid"
            format_ = lambda id: f" {id + 1}"
        else:
            path = ""
            format_ = lambda id: f" ({id + 1}-{self.residues[id]})"

        for residue_index in visited_residues_indices:
            path += format_(residue_index)
        return path

    def _get_transitions_prob_vector(self, residue_index: int, transition_probabilities_matrix: np.array) -> np.array:
        return transition_probabilities_matrix[residue_index, :]

    def _get_next_probability_matrix_and_selection_probability(self, preceding_residue: Union[None, int], current_residue: int, next_residue: int, current_matrix_index: int) -> Tuple[int, float]:

        # Ensure preceding_residue is valid
        if preceding_residue is None or preceding_residue in (current_residue, next_residue):
            valid_residues = [i for i in range(self.residues_range) if i not in (current_residue, next_residue)]
            preceding_residue = np.random.choice(valid_residues)

        # Get the last observed energy between preceding and current residue
        last_observed_energy_btw_preceding_current: float = self.interactions_matrices[current_matrix_index][preceding_residue, current_residue]

        # Calculate rounded energies and their probabilities
        rounded_energy_counts_btw_current_next: np.ndarray = np.round([matrix[current_residue, next_residue] 
                                                        for matrix in self.interactions_matrices],
                                                        decimals=self.interactions_precision_limit_decimals) # 1-D array
        
        # Get unique rounded interaction energies and their frequencies
        values_counts: Tuple[np.ndarray, np.ndarray] = np.unique(rounded_energy_counts_btw_current_next, return_counts=True) # 1-D arrays
        # unique interaction energies between the current and the next reisues, and their frequencies across the matrices
        unique_rounded_energies_btw_current_next, frequencies = values_counts # len(unique) == len(counts) -> True

        # Calculate probabilities
        probabilities: np.ndarray = frequencies / self.number_matrices

        # Draw an energy value based on the probability distribution
        drawn_energy: float = np.random.choice(unique_rounded_energies_btw_current_next, p=probabilities)

        # Select the matrix index where the energy matches
        selected_matrix_index: int = min(
            [(which_matrix, abs(matrix[preceding_residue, current_residue] - last_observed_energy_btw_preceding_current))
            for which_matrix, matrix in enumerate(self.interactions_matrices)
            if np.round(matrix[current_residue, next_residue], decimals=self.interactions_precision_limit_decimals) == drawn_energy],
            key=lambda matrix_difference_pair: matrix_difference_pair[1])[0]

        # Calculate matrix selection probability
        matrix_selection_probability: float = 1 / frequencies[np.where(unique_rounded_energies_btw_current_next == drawn_energy)][0]

        return selected_matrix_index, matrix_selection_probability

    def _generate_allosteric_signal_pathway(self, start: int, number_steps: Union[None, int] = None, target_residues: Union[None, Tuple[int]] = None) -> Tuple[list, float]:
        
        pathway: List[int] = [start]
        log_aggregated_probability: float = 0.0

        current_matrix_index: int = np.random.randint(0, self.number_matrices)
        preceding_residue: int = None
        current_residue: int = start
        next_residue: Union[int, None] = None

        for _ in range(number_steps):
            probability_vector: np.ndarray = self._get_transitions_prob_vector(current_residue, self.probabilities_matrices[current_matrix_index]) # 1-D array
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
    
    def _generate_multiple_pathways(self, num_pathways: int, start: int, number_steps: Union[None, int] = None, target_residues: Union[None, Tuple[int]] = None):
        pathway_probability_pairs = []
        for _ in range(num_pathways):
            pathway_probability_pairs.append(self._generate_allosteric_signal_pathway(start, number_steps, target_residues))
        return pathway_probability_pairs

    def create_pathways(self, start_residue: int, number_steps: Union[None, int] = None, target_residues: Union[None, Tuple[int]] = None,
                        number_pathways: int = 100, filter_out_improbable: bool = True, percentage_kept: float = 0.1):
        
        if number_steps is None:
            number_steps = self.residues_range
        
        if not (0.0 <= start_residue <= self.number_residues):
            raise ValueError(f"start_residue {start_residue} value is invalid; expected an integer value in [0.0, {self.number_residues}]")
        if not (0.0 <= number_steps <= self.number_residues):
            raise ValueError(f"number_steps {number_steps} value is invalid; expected an integer value in [0.0, {self.number_residues}]")
        if not (0.0 < percentage_kept <= 1.0):
            raise ValueError(f"percentage_kept {percentage_kept} value is invalid; expected a floating point value in (0.0, 1.0]")
        
        try:
            available_cores = os.cpu_count()

            # Calculate batch sizes for each core
            pathway_batch_size, residual_pathways = divmod(number_pathways, available_cores)
            pathway_batches = [pathway_batch_size + 1 if i < residual_pathways else pathway_batch_size for i in range(available_cores)]

            # Generate pathways in parallel with ProcessPoolExecutor
            generated_pathways = util.process_elementwise(in_parallel=False, Executor=ProcessPoolExecutor, max_workers=available_cores)(
                pathway_batches, self._generate_multiple_pathways, start_residue, number_steps, target_residues)

            # Flatten the list of lists of pathways
            generated_pathways = list(chain.from_iterable(generated_pathways))
            generated_pathways.sort(key=lambda x: x[1], reverse=True)

            if filter_out_improbable:
                number_kept = int(number_pathways * percentage_kept)
                most_probable_pathways = generated_pathways[:number_kept]
            else:
                most_probable_pathways = generated_pathways
            
            with open(self.output_file, "w") as output:
                header = f"""Generated allosteric pathways sorted from more probable to less probable (top to bottom)
                The following parameters were used:
                start_residue: {start_residue}
                number_steps: {number_steps}
                target_residues: {target_residues}
                number_pathways: {number_pathways}
                filter_out_improbable: {filter_out_improbable}
                percentage_kept: {percentage_kept}
                random_seed: {self.seed}
                """
                output.write(header + "\n")
                for index, pathway_and_probability in enumerate(most_probable_pathways, start=1):
                    pathway, _ = pathway_and_probability
                    output.write(f"{index}) {self._format_pathway(pathway)}\n")

            return self.output_file

        except Exception:
            raise

def main():
    pass


if __name__ == "__main__":
    main()
