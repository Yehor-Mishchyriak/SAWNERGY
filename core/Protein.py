#!AllostericPathwayAnalyzer/venv/bin/python3

# external imports
import os
from typing import Union, Tuple, List
from math import log, exp
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from itertools import chain
import numpy as np

# local imports
import core
from .util import normalize_vector, process_elementwise, init_error_handler_n_logger, generic_error_handler_n_logger, import_network_components


class Protein:
    """
    Represents a protein as a multidimensional electrostatic network with capabilities of allosteric communication analysis.
    
    This class provides methods to load and manage protein interaction networks, store matrices in shared memory,
    and generate allosteric pathways based on signal transition probabilities and interaction energies between residues.

    !Note: THE CLASS USES Multiprocessing TO ACHIEVE PARALLELISM

    Attributes:
        residues (tuple): an tuple of residue names, ordered so that each index corresponds to its residue name;
                        (can be thought of as a map from indices to residues, e.g. residues[219] -> TYR);
                        Note: it is 0-base-indexed.
        number_residues (int): the total number of residues in the protein, that is, nodes in the network.
        number_matrices (int): the total number of matrix-pairs (interaction & probability);
                        depends on the number_frames and batch_size parameters used during the network buidling process.
        seed (int): the seed value used for the operations involving probability, for example signal transition
                        from one node to the next; it is logged and can be used for replicating results.
        output_file (str): Path to the directory where the output pathways will be saved.
    """

    @init_error_handler_n_logger(core.protein_module_logger)
    def __init__(self, network_directory_path: str, interactions_precision_limit_decimals: int = 1, seed: Union[int, None] = None) -> None:
        """
        Initializes the Protein class by importing network components and setting up attributes.

        Args:
            network_directory_path (str): Path to the directory containing interaction and probability matrices, and the indices to residues map.
            interactions_precision_limit_decimals (int, optional): Precision (number of decimal places) for interaction energy calculations. Defaults to 1.
            seed (Union[int, None], optional): seed value used for the operations involving probability. Defaults to None.
        """
        # import the network components
        # NOTE: when working with a large number of matrices, one should store only a limited amount of them in memory;
        # Each 393x393 matrix (in case of p53) is ~2.4MB, so 100 matrices is ~240MB and 1,000 matrices is ~2.4GB, which isn't catostrophically large,
        # but if you are working with a bigger system than p53, the size of individual matrices will grow quadratically
        network_components: Tuple[Tuple] = import_network_components(network_directory_path)
        core.protein_module_logger.info(f"Successfully imported the network components")

        # set up public instance attributes
        self.residues: tuple = network_components[0]
        self.number_residues: int = len(self.residues)
        self.number_matrices: int = len(network_components[1])
        self.seed: int = self._set_seed(seed)
        self.output_file: str = os.path.join(core.create_output_dir(core.root_config["GLOBAL"]["output_directory_path"],
                                                                    core.root_config["Protein"]["output_directory_name"]),
                                                                    core.root_config["Protein"]["pathways_file_name"])

        # set up private instance attributes
        self._residues_range: int = self.number_residues
        self._interaction_matrices_shared_memory = Protein._store_matrices_in_shared_memory(network_components[1])
        self._probability_matrices_shared_memory = Protein._store_matrices_in_shared_memory(network_components[2])
        self._memory_cleaned_up = False
        self._matrix_size = network_components[1][0].nbytes
        self._matrix_shape = network_components[1][0].shape
        self._matrix_dtype = network_components[1][0].dtype
        self._interactions_precision_limit_decimals: int = interactions_precision_limit_decimals

        core.protein_module_logger.info(f"Successfully initialized the {self.__class__.__name__} class with:
                                        {self.number_residues} residues, {self.number_matrices} matrices, and seed = {self.seed}.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.memory_cleanup()

    def __del__(self):
        self.memory_cleanup()

    #######################
    # INIT HELPER METHODS #
    #######################

    @staticmethod
    @generic_error_handler_n_logger(core.protein_module_logger)
    def _store_matrices_in_shared_memory(matrices: Tuple[np.ndarray]) -> shared_memory.SharedMemory:
        """
        Stores a sequence of matrices in shared memory for efficient parallel processing.

        Args:
            matrices (Tuple[np.ndarray]): A tuple of numpy matrices to be stored.

        Returns:
            shared_memory.SharedMemory: Shared memory block containing all the matrices.
        """
        total_size = sum(matrix.nbytes for matrix in matrices)
        # create shared memory for the matrices
        core.protein_module_logger.info(f"Creating shared memory for matrices, total size: {total_size} bytes.")
        shm = shared_memory.SharedMemory(create=True, size=total_size)
        # copy the matrices to the shared memory block
        offset_size = 0
        for matrix in matrices:
            matrix_size = matrix.nbytes
            mapped_matrix = np.ndarray(matrix.shape, dtype=matrix.dtype, buffer=shm.buf[offset_size:offset_size + matrix_size])
            np.copyto(mapped_matrix, matrix)  # copy data into shared memory
            offset_size += matrix_size
        core.protein_module_logger.info(f"Successfully stored {len(matrices)} matrices in shared memory.")
        return shm
    
    @generic_error_handler_n_logger(core.protein_module_logger)
    def _set_seed(self, seed: Union[int, None] = None) -> int:
        """
        Sets the seed for random number generation and stores it in the instance.

        Args:
            seed (Union[int, None], optional): Seed value. If None, a random seed is generated. Defaults to None.

        Returns:
            int: The seed value used.
        """
        seed = np.random.randint(0, 2**32 - 1) if seed is None else seed
        self.seed = seed
        np.random.seed(seed)
        return seed
    
    #######################
    # MEMORY INTERACTIONS #
    #######################

    @generic_error_handler_n_logger(core.protein_module_logger)
    def _get_interaction_matrix(self, index: int) -> np.ndarray:
        """
        Retrieves a specific interaction matrix from shared memory.

        Args:
            index (int): Index of the interaction matrix to retrieve.

        Returns:
            np.ndarray: The interaction matrix.
        """
        # access the memory block
        memory_block = self._interaction_matrices_shared_memory
        # calculate the offset size
        offset_size = self._matrix_size * index
        # retrieve the required slice of the memory buffer
        data = memory_block.buf[offset_size:offset_size + self._matrix_size]
        # reconstruct and return the matrix based on the data from the slice
        matrix = np.ndarray(self._matrix_shape, dtype=self._matrix_dtype, buffer=data)
        return matrix

    @generic_error_handler_n_logger(core.protein_module_logger)
    def _get_probability_matrix(self, index: int) -> np.ndarray:
        """
        Retrieves a specific probability matrix from shared memory.

        Args:
            index (int): Index of the probability matrix to retrieve.

        Returns:
            np.ndarray: The probability matrix.
        """
        # access the memory block
        memory_block = self._probability_matrices_shared_memory
        # calculate the offset size
        offset_size = self._matrix_size * index
        # retrieve the required slice of the memory buffer
        data = memory_block.buf[offset_size:offset_size + self._matrix_size]
        # reconstruct and return the matrix based on the data from the slice
        matrix = np.ndarray(self._matrix_shape, dtype=self._matrix_dtype, buffer=data)
        return matrix

    @generic_error_handler_n_logger(core.protein_module_logger)
    def _get_all_interaction_matrices(self):
        """
        Retrieves all interaction matrices from shared memory.

        Returns:
            List[np.ndarray]: List of all interaction matrices.
        """
        return [self._get_interaction_matrix(i) for i in range(self.number_matrices)]

    @generic_error_handler_n_logger(core.protein_module_logger)
    def _get_all_probability_matrices(self):
        """
        Retrieves all probability matrices from shared memory.

        Returns:
            List[np.ndarray]: List of all probability matrices.
        """
        return [self._get_probability_matrix(i) for i in range(self.number_matrices)]

    @generic_error_handler_n_logger(core.protein_module_logger)
    def _get_transitions_prob_vector(self, residue_index: int, transition_probabilities_matrix: np.array) -> np.array:
        """
        Retrieves the transition probabilities vector (distribution) for a given residue index.

        Args:
            residue_index (int): Index of the residue.
            transition_probabilities_matrix (np.ndarray): The probability matrix.

        Returns:
            np.ndarray: Transition probabilities vector for the residue.
        """
        return transition_probabilities_matrix[residue_index, :].copy()

    ######################################
    # PATHWAYS GENERATION AND FORMATTING #
    ######################################

    @generic_error_handler_n_logger(core.protein_module_logger)
    def _get_next_probability_matrix_and_selection_probability(self, preceding_residue: Union[None, int], current_residue: int, next_residue: int, current_matrix_index: int) -> Tuple[int, float]:
        """
        Selects the next probability matrix and calculates its selection probability based on current and preceding residues.

        Args:
            preceding_residue (Union[None, int]): Index of the preceding residue.
            current_residue (int): Index of the current residue.
            next_residue (int): Index of the next residue.
            current_matrix_index (int): Index of the current matrix.

        Returns:
            Tuple[int, float]: Selected matrix index and its selection probability.
        """
        core.protein_module_logger.debug(f"Selecting the next probability matrix based on:
                                         preceding_residue: {preceding_residue},
                                         current_residue: {current_residue},
                                         current_matrix_index: {current_matrix_index}")
        # ensure preceding_residue is valid
        if preceding_residue is None or preceding_residue in (current_residue, next_residue):
            valid_residues = [i for i in range(self._residues_range) if i not in (current_residue, next_residue)]
            preceding_residue = np.random.choice(valid_residues)

        # get the last observed energy between preceding and current residue
        last_observed_energy_btw_preceding_current: float = self._get_interaction_matrix(current_matrix_index)[preceding_residue, current_residue]
        core.protein_module_logger.debug(f"last_observed_energy_btw_preceding_current: {last_observed_energy_btw_preceding_current}")
        # calculate rounded energies and their probabilities
        rounded_energy_counts_btw_current_next: np.ndarray = np.round([matrix[current_residue, next_residue] 
                                                        for matrix in self._get_all_interaction_matrices()],
                                                        decimals=self._interactions_precision_limit_decimals) # 1-D array
        
        # get unique rounded interaction energies and their frequencies
        values_counts: Tuple[np.ndarray] = np.unique(rounded_energy_counts_btw_current_next, return_counts=True) # 1-D arrays
        # unique interaction energies between the current and the next reisues, and their frequencies across the matrices
        unique_rounded_energies_btw_current_next, frequencies = values_counts # len(unique) == len(counts) -> True
        core.protein_module_logger.debug(f"unique_rounded_energies_btw_current_next and frequencies:
                                         {zip(unique_rounded_energies_btw_current_next, frequencies)}")
        # calculate probabilities
        probabilities: np.ndarray = frequencies / self.number_matrices

        # draw an energy value based on the probability distribution
        drawn_energy: float = np.random.choice(unique_rounded_energies_btw_current_next, p=probabilities)
        core.protein_module_logger.debug(f"The drawn energies: {drawn_energy}")
        # select the matrix index where the energy matches
        selected_matrix_index: int = min(
            [(which_matrix, abs(matrix[preceding_residue, current_residue] - last_observed_energy_btw_preceding_current))
            for which_matrix, matrix in enumerate(self._get_all_interaction_matrices())
            if np.round(matrix[current_residue, next_residue], decimals=self._interactions_precision_limit_decimals) == drawn_energy],
            key=lambda matrix_difference_pair: matrix_difference_pair[1])[0]

        # calculate matrix selection probability
        matrix_selection_probability: float = 1 / frequencies[np.where(unique_rounded_energies_btw_current_next == drawn_energy)][0]

        core.protein_module_logger.debug(f"selected_matrix_index, matrix_selection_probability: {selected_matrix_index}, {matrix_selection_probability}")

        return selected_matrix_index, matrix_selection_probability

    @generic_error_handler_n_logger(core.protein_module_logger)
    def _generate_allosteric_signal_pathway(self, start: int, number_steps: Union[None, int] = None, target_residues: Union[None, Tuple[int]] = None) -> Tuple[list, float]:
        """
        Generates a single allosteric signal pathway starting from a given residue.

        Args:
            start (int): Index of the starting residue.
            number_steps (Union[None, int], optional): Number of steps in the pathway. Defaults to None.
            target_residues (Union[None, Tuple[int]], optional): Target residues to stop the pathway generation. Defaults to None.

        Returns:
            Tuple[List[int], float]: Generated pathway and its aggregated probability.
        """
        pathway: List[int] = [start]
        log_aggregated_probability: float = 0.0

        current_matrix_index: int = np.random.randint(0, self.number_matrices)
        preceding_residue: int = None
        current_residue: int = start
        next_residue: Union[int, None] = None

        for _ in range(number_steps):
            core.protein_module_logger.debug(f"current_matrix_index: {current_matrix_index}, preceding_residue: {preceding_residue}, current_residue: {current_residue}, next_residue: {next_residue}")
            probability_vector: np.ndarray = self._get_transitions_prob_vector(current_residue, self._get_probability_matrix(current_matrix_index)) # 1-D array
            probability_vector[pathway] = 0.0  # Avoid loops by setting already visited residues to 0
            core.protein_module_logger.debug(f"Current pathway: {pathway}")
            probability_vector = normalize_vector(probability_vector)
            core.protein_module_logger.debug(f"Current probability vector for moving from {current_residue}: {probability_vector}")
            next_residue = np.random.choice(range(0, self._residues_range), p=probability_vector)
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
    
    @generic_error_handler_n_logger(core.protein_module_logger)
    def _generate_multiple_pathways(self, num_pathways: int, start: int, number_steps: Union[None, int] = None, target_residues: Union[None, Tuple[int]] = None):
        """
        Generates multiple allosteric signal pathways.

        Args:
            num_pathways (int): Number of pathways to generate.
            start (int): Index of the starting residue.
            number_steps (Union[None, int], optional): Number of steps in each pathway. Defaults to None.
            target_residues (Union[None, Tuple[int]], optional): Target residues to stop the pathway generation. Defaults to None.

        Returns:
            List[Tuple[List[int], float]]: List of generated pathways and their probabilities.
        """
        pathway_probability_pairs = []
        for _ in range(num_pathways):
            pathway_probability_pairs.append(self._generate_allosteric_signal_pathway(start, number_steps, target_residues))
        return pathway_probability_pairs

    @generic_error_handler_n_logger(core.protein_module_logger)
    def _format_pathway(self, visited_residues_indices: list, VMD_compatible: bool = True) -> str:
        """
        Formats a pathway for output.

        Args:
            visited_residues_indices (List[int]): List of residue indices in the pathway.
            VMD_compatible (bool, optional): Whether the output should be compatible with VMD. Defaults to True.

        Returns:
            str: Formatted pathway.
        """
        if VMD_compatible:
            path = "resid"
            format_ = lambda id: f" {id + 1}"
        else:
            path = ""
            format_ = lambda id: f" ({id + 1}-{self.residues[id]})"

        for residue_index in visited_residues_indices:
            path += format_(residue_index)
        return path

    ##################
    # PUBLIC METHODS #
    ##################

    @generic_error_handler_n_logger(core.protein_module_logger)
    def create_pathways(self, start_residue: int, number_steps: Union[None, int] = None, target_residues: Union[None, Tuple[int]] = None,
                        number_pathways: int = 100, filter_out_improbable: bool = True, percentage_kept: float = 0.1, in_parallel=True):
        """
        Generates and formats allosteric pathways, saving the results to a file.

        Args:
            start_residue (int): Index of the starting residue.
            number_steps (Union[None, int], optional): Number of steps in each pathway. Defaults to None.
            target_residues (Union[None, Tuple[int]], optional): Target residues for the pathways. Defaults to None.
            number_pathways (int, optional): Total number of pathways to generate. Defaults to 100.
            filter_out_improbable (bool, optional): Whether to filter out improbable pathways. Defaults to True.
            percentage_kept (float, optional): Percentage of the most probable pathways to keep. Defaults to 0.1.

        Returns:
            str: Path to the file containing the generated pathways.
        """
        available_cores = os.cpu_count()

        # calculate batch sizes for each core
        pathway_batch_size, residual_pathways = divmod(number_pathways, available_cores)
        pathway_batches = [pathway_batch_size + 1 if i < residual_pathways else pathway_batch_size for i in range(available_cores)]

        if in_parallel:
            core.protein_module_logger.info(f"Generating {len(pathway_batches)} pathway batches in parallel; Each batch is {pathway_batch_size} pathways big")

            try:
                generated_pathways = process_elementwise(in_parallel=True, Executor=ProcessPoolExecutor, max_workers=available_cores)(
                    pathway_batches, self._generate_multiple_pathways, start_residue, number_steps, target_residues)
                
            except RuntimeError as e:
                if __name__ != "__main__":
                    core.protein_module_logger.warning(f"Parallel processing has failed likely due to platform-specific issues;"
                                                             f"to allow {self.__class__.__name__} to use parallel processing,
                                                             execute it as the main module -- not an imported module."
                                                             f"Error message: {e}")
                else:
                    core.protein_module_logger.warning(f"Parallel processing has failed due to an unexpected error")

                core.protein_module_logger.info("Falling back to sequential processing")
                generated_pathways = process_elementwise(in_parallel=False)(
                pathway_batches, self._generate_multiple_pathways, start_residue, number_steps, target_residues)
                
        else:
            core.protein_module_logger.info(f"Generating {len(pathway_batches)} pathway batches sequentially; Each batch is {pathway_batch_size} pathways big")
            generated_pathways = process_elementwise(in_parallel=False)(
                pathway_batches, self._generate_multiple_pathways, start_residue, number_steps, target_residues)

        core.protein_module_logger.info(f"Successfully generated all the pathways")
        # flatten the list of lists of pathways
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
    
    @generic_error_handler_n_logger(core.protein_module_logger)
    def memory_cleanup(self):
        """
        Cleans up shared memory used by the class.
        """
        core.protein_module_logger.info(f"Attempting to clean up the shared memory space")

        if self._memory_cleaned_up:
            core.protein_module_logger.info(f"The shared memory space was already cleaned up")
            return
        # perform cleanup
        self._interaction_matrices_shared_memory.close()
        self._interaction_matrices_shared_memory.unlink()
        self._probability_matrices_shared_memory.close()
        self._probability_matrices_shared_memory.unlink()
        
        # set the flag to indicate cleanup is done
        self._memory_cleaned_up = True
        core.protein_module_logger.info(f"Successully cleaned up the shared memory space")


def main():
    pass


if __name__ == "__main__":
    main()
