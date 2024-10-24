import os
import numpy as np
from typing import Union, Tuple, List
from math import log, exp
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
import newUtil as util # this is temporary
from itertools import chain

'''
PLEASE NOTE: This project is ready and works in the context of the HPC class, however as a component of my research project,
it is incomplete yet. There are a lot of things that will be added, such as error handling, logging, extra comments, docstrings, and overall code refactoring
to ensure adherence to PEP-8 and intuitive structure as well as consistent variable names.
'''

root_config = util.load_json_config("/home/yehor/research_project/AllostericPathwayAnalyzer/configs/root.json")
# logger = util.set_up_logging("AllostericPathwayAnalyzer/configs/logging.json", "network_construction_module")

class Protein:

    def __init__(self, network_directory_path: str, interactions_precision_limit_decimals: int = 1, seed: Union[int, None] = None) -> None:

        network_components: Tuple[Tuple] = util.import_network_components(network_directory_path)

        self.residues = network_components[0]
        self.number_residues: int = len(self.residues)
        self.residues_range: int = self.number_residues

        self.interaction_matrices_shared_memory = Protein._store_matrices_in_shared_memory(network_components[1])
        self.probability_matrices_shared_memory = Protein._store_matrices_in_shared_memory(network_components[2])
        self.matrix_size = network_components[1][0].nbytes
        self.matrix_shape = network_components[1][0].shape
        self.matrix_dtype = network_components[1][0].dtype
        self.number_matrices: int = len(network_components[1])

        self.interactions_precision_limit_decimals: int = interactions_precision_limit_decimals

        self.seed: int = self._set_seed(seed)
        self.output_file: str = os.path.join(util.create_output_dir(root_config["GLOBAL"]["output_directory_path"],
                                                                    root_config["Protein"]["output_directory_name"]),
                                                                    root_config["Protein"]["pathways_file_name"])

    @staticmethod
    def _store_matrices_in_shared_memory(matrices: Tuple[np.ndarray]) -> shared_memory.SharedMemory:
        total_size = sum(matrix.nbytes for matrix in matrices)
        # Create shared memory for the matrices
        shm = shared_memory.SharedMemory(create=True, size=total_size)
        # Copy the matrices to the shared memory block
        offset_size = 0
        for matrix in matrices:
            matrix_size = matrix.nbytes
            mapped_matrix = np.ndarray(matrix.shape, dtype=matrix.dtype, buffer=shm.buf[offset_size:offset_size + matrix_size])
            np.copyto(mapped_matrix, matrix)  # Copy data into shared memory
            offset_size += matrix_size
        return shm

    def _get_interaction_matrix(self, index: int) -> np.ndarray:
        # access the memory block
        memory_block = self.interaction_matrices_shared_memory
        # calculate the offset size
        offset_size = self.matrix_size * index
        # retrieve the required slice of the memory buffer
        data = memory_block.buf[offset_size:offset_size + self.matrix_size]
        # reconstruct and return the matrix based on the data from the slice
        matrix = np.ndarray(self.matrix_shape, dtype=self.matrix_dtype, buffer=data)
        return matrix

    def _get_probability_matrix(self, index: int) -> np.ndarray:
        # access the memory block
        memory_block = self.probability_matrices_shared_memory
        # calculate the offset size
        offset_size = self.matrix_size * index
        # retrieve the required slice of the memory buffer
        data = memory_block.buf[offset_size:offset_size + self.matrix_size]
        # reconstruct and return the matrix based on the data from the slice
        matrix = np.ndarray(self.matrix_shape, dtype=self.matrix_dtype, buffer=data)
        return matrix

    def _get_all_interaction_matrices(self):
        return [self._get_interaction_matrix(i) for i in range(self.number_matrices)]

    def _get_all_probability_matrices(self):
        return [self._get_probability_matrix(i) for i in range(self.number_matrices)]

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
        return transition_probabilities_matrix[residue_index, :].copy()

    def _get_next_probability_matrix_and_selection_probability(self, preceding_residue: Union[None, int], current_residue: int, next_residue: int, current_matrix_index: int) -> Tuple[int, float]:

        # Ensure preceding_residue is valid
        if preceding_residue is None or preceding_residue in (current_residue, next_residue):
            valid_residues = [i for i in range(self.residues_range) if i not in (current_residue, next_residue)]
            preceding_residue = np.random.choice(valid_residues)

        # Get the last observed energy between preceding and current residue
        last_observed_energy_btw_preceding_current: float = self._get_interaction_matrix(current_matrix_index)[preceding_residue, current_residue]

        # Calculate rounded energies and their probabilities
        rounded_energy_counts_btw_current_next: np.ndarray = np.round([matrix[current_residue, next_residue] 
                                                        for matrix in self._get_all_interaction_matrices()],
                                                        decimals=self.interactions_precision_limit_decimals) # 1-D array
        
        # Get unique rounded interaction energies and their frequencies
        values_counts: Tuple[np.ndarray] = np.unique(rounded_energy_counts_btw_current_next, return_counts=True) # 1-D arrays
        # unique interaction energies between the current and the next reisues, and their frequencies across the matrices
        unique_rounded_energies_btw_current_next, frequencies = values_counts # len(unique) == len(counts) -> True

        # Calculate probabilities
        probabilities: np.ndarray = frequencies / self.number_matrices

        # Draw an energy value based on the probability distribution
        drawn_energy: float = np.random.choice(unique_rounded_energies_btw_current_next, p=probabilities)

        # Select the matrix index where the energy matches
        selected_matrix_index: int = min(
            [(which_matrix, abs(matrix[preceding_residue, current_residue] - last_observed_energy_btw_preceding_current))
            for which_matrix, matrix in enumerate(self._get_all_interaction_matrices())
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
            probability_vector: np.ndarray = self._get_transitions_prob_vector(current_residue, self._get_probability_matrix(current_matrix_index)) # 1-D array
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

        available_cores = os.cpu_count()

        # Calculate batch sizes for each core
        pathway_batch_size, residual_pathways = divmod(number_pathways, available_cores)
        pathway_batches = [pathway_batch_size + 1 if i < residual_pathways else pathway_batch_size for i in range(available_cores)]

        # Generate pathways in parallel with ProcessPoolExecutor
        generated_pathways = util.process_elementwise(in_parallel=True, Executor=ProcessPoolExecutor, max_workers=available_cores)(
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
    
    # IMPORTANT: RUN AFTER YOU ARE DONE USING THE CLASS
    def memory_cleanup(self):
        # might wrap it with a class destructor down the road. We'll see
        self.interaction_matrices_shared_memory.close()
        self.interaction_matrices_shared_memory.unlink()
        self.probability_matrices_shared_memory.close()
        self.probability_matrices_shared_memory.unlink()


def main():
    pass


if __name__ == "__main__":
    main()
