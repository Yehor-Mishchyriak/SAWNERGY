# external imports
import os
from typing import Tuple, List, Optional, Dict, Any, Type
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from itertools import chain
import numpy as np
from types import TracebackType

# local imports
from . import pkg_globals
from . import _util

# TODO: 
# [] fix the docstrings
# [] ensure the indexing is correct
# [] investigate the matrices

class Protein:
    """
    Represents a Protein with its network components and provides methods for generating allosteric signal pathways.

    The Protein class loads network components (residues, interaction matrices, and probability matrices) from a
    specified directory using a given configuration. It stores the matrices in shared memory and provides methods
    to generate, format, and write pathways based on transition probabilities.
    """

    def __init__(self, network_directory_path: str, interactions_precision_limit_decimals: int = 1) -> None:
        """
        Initialize a Protein instance by loading network components from a specified directory.

        The network components (residues, interaction matrices, and probability matrices) are imported using
        the default configuration from pkg_globals. Interaction and probability matrices are stored in shared memory
        for efficient access.

        Args:
            network_directory_path (str): Path to the directory containing network component files.
            interactions_precision_limit_decimals (int, optional): Number of decimals used when rounding interaction energies.
                Defaults to 1.
        """
        self.global_config = None
        self.cls_config = None
        self.set_config(pkg_globals.default_config)
        network_components: Tuple[
            Tuple[Any, ...], Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]
        ] = _util.import_network_components(network_directory_path, self.global_config)

        # Set up public instance attributes.
        self.residues: Tuple[Any, ...] = network_components[0]
        self.number_residues: int = len(self.residues)
        self.number_matrices: int = len(network_components[1])

        # Set up private instance attributes.
        self._residues_range: int = self.number_residues
        self._memory_cleaned_up: bool = False
        self._interaction_matrices_shared_memory: shared_memory.SharedMemory = Protein._store_matrices_in_shared_memory(network_components[1])
        self._probability_matrices_shared_memory: shared_memory.SharedMemory = Protein._store_matrices_in_shared_memory(network_components[2])
        self._matrix_size: int = network_components[1][0].nbytes
        self._matrix_shape: Tuple[int, ...] = network_components[1][0].shape
        self._matrix_dtype: np.dtype = network_components[1][0].dtype
        self._interactions_precision_limit_decimals: int = interactions_precision_limit_decimals

    def __enter__(self) -> "Protein":
        """
        Enter the runtime context for the Protein instance.

        Returns:
            Protein: The current Protein instance.
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType]
    ) -> None:
        """
        Exit the runtime context and perform cleanup.

        This method ensures that shared memory resources are cleaned up when exiting the context.

        Args:
            exc_type (Optional[Type[BaseException]]): The type of exception raised, if any.
            exc_value (Optional[BaseException]): The exception instance raised, if any.
            traceback (Optional[TracebackType]): Traceback information, if an exception occurred.
        """
        self.memory_cleanup()

    def __repr__(self) -> str:
        """
        Return a string representation of the Protein instance.

        Returns:
            str: A string that includes the number of residues, number of matrices, and the pathways file name from the configuration.
        """
        return f"{self.__class__.__name__}(number_residues={self.number_residues}, number_matrices={self.number_matrices})"

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the global and ToMatricesConverter-specific configuration.

        The configuration must include:
            - ToMatricesConverter
                - 'output_directory_name'
                - 'matrices_directory_name'
                - 'interactions_matrix_name'
                - 'probabilities_matrix_name'
                - 'id_to_res_map_name'
            - Protein
                - 'output_directory_name'
                - 'pathways_file_name'
                - 'pathways_file_header'

        Args:
            config (dict): A configuration dictionary
        
        Raises:
            ValueError: If the required configuration keys are missing
        """
        if "ToMatricesConverter" not in config:
            raise ValueError(f"Invalid config: missing ToMatricesConverter sub-config")
        
        tmc_sub_config = config["ToMatricesConverter"]
        if "output_directory_name" not in tmc_sub_config:
            raise ValueError(f"Invalid ToMatricesConverter sub-config: missing output_directory_name field")
        if "matrices_directory_name" not in tmc_sub_config:
            raise ValueError(f"Invalid ToMatricesConverter sub-config: missing matrices_directory_name field")
        if "interactions_matrix_name" not in tmc_sub_config:
            raise ValueError(f"Invalid ToMatricesConverter sub-config: missing interactions_matrix_name field")
        if "probabilities_matrix_name" not in tmc_sub_config:
            raise ValueError(f"Invalid ToMatricesConverter sub-config: missing probabilities_matrix_name field")
        if "id_to_res_map_name" not in tmc_sub_config:
            raise ValueError(f"Invalid ToMatricesConverter sub-config: missing id_to_res_map_name field")

        if self.__class__.__name__ not in config:
            raise ValueError(f"Invalid config: missing {self.__class__.__name__} sub-config")

        protein_sub_config = config[self.__class__.__name__]
        if "output_directory_name" not in protein_sub_config:
            raise ValueError(f"Invalid {self.__class__.__name__} sub-config: missing output_directory_name field")
        if "pathways_file_name" not in protein_sub_config:
            raise ValueError(f"Invalid {self.__class__.__name__} sub-config: missing pathways_file_name field")
        if "pathways_file_header" not in protein_sub_config:
            raise ValueError(f"Invalid {self.__class__.__name__} sub-config: missing pathways_file_header field")
        
        self.global_config = config
        self.cls_config = config[self.__class__.__name__]

    #######################
    # INIT HELPER METHODS #
    #######################

    @staticmethod
    def _store_matrices_in_shared_memory(matrices: Tuple[np.ndarray, ...]) -> shared_memory.SharedMemory:
        """
        Store a tuple of numpy arrays in a contiguous shared memory block.

        The method calculates the total size required and copies each matrix into the shared memory buffer.

        Args:
            matrices (Tuple[np.ndarray, ...]): Tuple of numpy arrays to be stored.

        Returns:
            shared_memory.SharedMemory: A shared memory object containing the matrices.
        """
        total_size: int = sum(matrix.nbytes for matrix in matrices)
        # Create shared memory for the matrices.
        shm: shared_memory.SharedMemory = shared_memory.SharedMemory(create=True, size=total_size)
        # Copy the matrices into the shared memory block.
        offset_size: int = 0
        for matrix in matrices:
            matrix_size: int = matrix.nbytes
            mapped_matrix: np.ndarray = np.ndarray(matrix.shape, dtype=matrix.dtype, buffer=shm.buf[offset_size:offset_size + matrix_size])
            np.copyto(mapped_matrix, matrix)  # Copy data into shared memory.
            offset_size += matrix_size
        return shm

    #######################
    # MEMORY INTERACTIONS #
    #######################

    def _get_interaction_matrix(self, index: int) -> np.ndarray:
        """
        Retrieve an interaction matrix from shared memory by its index.

        Args:
            index (int): The index of the interaction matrix to retrieve.

        Returns:
            np.ndarray: The specified interaction matrix.
        """
        memory_block: shared_memory.SharedMemory = self._interaction_matrices_shared_memory
        offset_size: int = self._matrix_size * index
        data = memory_block.buf[offset_size:offset_size + self._matrix_size]
        matrix: np.ndarray = np.ndarray(self._matrix_shape, dtype=self._matrix_dtype, buffer=data)
        return matrix

    def _get_probability_matrix(self, index: int) -> np.ndarray:
        """
        Retrieve a probability matrix from shared memory by its index.

        Args:
            index (int): The index of the probability matrix to retrieve.

        Returns:
            np.ndarray: The specified probability matrix.
        """
        memory_block: shared_memory.SharedMemory = self._probability_matrices_shared_memory
        offset_size: int = self._matrix_size * index
        data = memory_block.buf[offset_size:offset_size + self._matrix_size]
        matrix: np.ndarray = np.ndarray(self._matrix_shape, dtype=self._matrix_dtype, buffer=data)
        return matrix

    def _get_all_interaction_matrices(self) -> List[np.ndarray]:
        """
        Retrieve all interaction matrices stored in shared memory.

        Returns:
            List[np.ndarray]: A list containing all interaction matrices.
        """
        return [self._get_interaction_matrix(i) for i in range(self.number_matrices)]

    def _get_all_probability_matrices(self) -> List[np.ndarray]:
        """
        Retrieve all probability matrices stored in shared memory.

        Returns:
            List[np.ndarray]: A list containing all probability matrices.
        """
        return [self._get_probability_matrix(i) for i in range(self.number_matrices)]

    def _get_transitions_prob_vector(self, residue_index: int, transition_probabilities_matrix: np.ndarray) -> np.ndarray:
        """
        Get the transition probability vector for a specific residue.

        Args:
            residue_index (int): The index of the residue.
            transition_probabilities_matrix (np.ndarray): The matrix containing transition probabilities.

        Returns:
            np.ndarray: A copy of the probability vector for the specified residue.
        """
        return transition_probabilities_matrix[residue_index, :].copy()

    ######################################
    # PATHWAYS GENERATION AND FORMATTING #
    ######################################

    def _get_next_probability_matrix_and_selection_probability(
        self, 
        preceding_residue: Optional[int], 
        current_residue: int, 
        next_residue: int, 
        current_matrix_index: int
    ) -> Tuple[int, float]:
        """
        Determine the next probability matrix index and its selection probability based on energy differences.

        This method calculates rounded energies from all interaction matrices, determines a probability distribution,
        and selects the matrix index that best matches the observed energy difference.

        Args:
            preceding_residue (Optional[int]): Index of the preceding residue. If None or invalid, a random valid residue is chosen.
            current_residue (int): Index of the current residue.
            next_residue (int): Index of the next residue.
            current_matrix_index (int): Index of the current interaction matrix.

        Returns:
            Tuple[int, float]: A tuple containing:
                - The selected matrix index.
                - The probability of selecting that matrix.
        """
        if preceding_residue is None or preceding_residue in (current_residue, next_residue):
            valid_residues: List[int] = [i for i in range(self._residues_range) if i not in (current_residue, next_residue)]
            # Randomly choose a valid residue when the preceding residue is invalid.
            preceding_residue = np.random.choice(valid_residues)

        # Get the last observed energy between the preceding and current residue.
        last_observed_energy_btw_preceding_current: float = self._get_interaction_matrix(current_matrix_index)[preceding_residue, current_residue]
        
        # Round the energies between the current and next residue across all matrices.
        rounded_energy_counts_btw_current_next: np.ndarray = np.round(
            [matrix[current_residue, next_residue] for matrix in self._get_all_interaction_matrices()],
            decimals=self._interactions_precision_limit_decimals
        )
        
        # Get unique rounded energies and their frequencies.
        values_counts: Tuple[np.ndarray, np.ndarray] = np.unique(rounded_energy_counts_btw_current_next, return_counts=True)
        unique_rounded_energies_btw_current_next, frequencies = values_counts

        # Calculate probabilities for each unique energy.
        probabilities: np.ndarray = frequencies / self.number_matrices

        # Draw an energy value based on the probability distribution.
        drawn_energy: float = np.random.choice(unique_rounded_energies_btw_current_next, p=probabilities)
        
        # Select the matrix index where the energy matches best.
        selected_matrix_index: int = min(
            [
                (which_matrix, abs(matrix[preceding_residue, current_residue] - last_observed_energy_btw_preceding_current))
                for which_matrix, matrix in enumerate(self._get_all_interaction_matrices())
                if np.round(matrix[current_residue, next_residue], decimals=self._interactions_precision_limit_decimals) == drawn_energy
            ],
            key=lambda matrix_difference_pair: matrix_difference_pair[1]
        )[0]

        # Calculate matrix selection probability.
        matrix_selection_probability: float = 1 / frequencies[np.where(unique_rounded_energies_btw_current_next == drawn_energy)][0]

        return selected_matrix_index, matrix_selection_probability

    def _generate_allosteric_signal_pathway(
        self, 
        start: int, 
        number_steps: int, 
        target_residues: Optional[Tuple[int, ...]] = None
    ) -> Tuple[List[int], float]:
        """
        Generate an allosteric signal pathway starting from a specified residue.

        The pathway is generated by iteratively selecting the next residue based on the transition probabilities
        from the probability matrices. The method calculates the aggregated probability of the pathway as the product
        of individual selection probabilities.

        Args:
            start (int): The starting residue index.
            number_steps (int): Maximum number of steps for the pathway.
            target_residues (Optional[Tuple[int, ...]]): Optional tuple of target residue indices that, if reached, will stop pathway generation.

        Returns:
            Tuple[List[int], float]: A tuple containing:
                - A list of residue indices representing the pathway.
                - The aggregated probability of the pathway.
        """
        pathway: List[int] = [start]
        log_aggregated_probability: float = 0.0

        current_matrix_index: int = np.random.randint(0, self.number_matrices)
        preceding_residue: Optional[int] = None
        current_residue: int = start
        next_residue: Optional[int] = None

        for _ in range(number_steps):
            # Get the transition probability vector for the current residue.
            probability_vector: np.ndarray = self._get_transitions_prob_vector(
                current_residue, self._get_probability_matrix(current_matrix_index)
            )
            # Avoid revisiting residues by zeroing out probabilities for already visited indices.
            probability_vector[pathway] = 0.0
            probability_vector = _util.normalize_vector(probability_vector)
            next_residue = np.random.choice(range(0, self._residues_range), p=probability_vector)
            residue_selection_probability: float = probability_vector[next_residue]

            next_matrix_index, matrix_selection_probability = self._get_next_probability_matrix_and_selection_probability(
                preceding_residue, current_residue, next_residue, current_matrix_index
            )

            pathway.append(next_residue)
            log_aggregated_probability += np.log(residue_selection_probability) + np.log(matrix_selection_probability)

            if target_residues is not None and next_residue in target_residues:
                break
            
            current_matrix_index = next_matrix_index
            preceding_residue = current_residue
            current_residue = next_residue
            next_residue = None

        aggregated_probability: float = np.exp(log_aggregated_probability)

        return pathway, aggregated_probability

    def _generate_multiple_pathways(
        self, 
        num_pathways: int, 
        start: int, 
        number_steps: int, 
        target_residues: Optional[Tuple[int, ...]] = None
    ) -> List[Tuple[List[int], float]]:
        """
        Generate multiple allosteric signal pathways.

        Args:
            num_pathways (int): The number of pathways to generate.
            start (int): The starting residue index for each pathway.
            number_steps (int): Maximum number of steps for each pathway.
            target_residues (Optional[Tuple[int, ...]]): Optional tuple of target residue indices that can terminate a pathway.

        Returns:
            List[Tuple[List[int], float]]: A list of tuples where each tuple contains a pathway (list of residue indices)
            and its aggregated probability.
        """
        pathway_probability_pairs: List[Tuple[List[int], float]] = []
        for _ in range(num_pathways):
            pathway_probability_pairs.append(self._generate_allosteric_signal_pathway(start, number_steps, target_residues))
        return pathway_probability_pairs

    def _format_pathway(self, visited_residues_indices: List[int], VMD_compatible: bool = True) -> str:
        """
        Format a pathway into a string representation.

        Depending on the VMD_compatible flag, the pathway is formatted differently:
          - If True, the pathway is formatted as residue numbers (starting with "resid").
          - If False, each residue is represented by its index and associated residue label.

        Args:
            visited_residues_indices (List[int]): A list of residue indices forming the pathway.
            VMD_compatible (bool, optional): Whether to format the pathway for VMD. Defaults to True.

        Returns:
            str: The formatted pathway string.
        """
        if VMD_compatible:
            path: str = "resid"
            format_ = lambda id: f" {id + 1}" # adding 1 due to zero-indexing under the hood
        else:
            path = ""
            format_ = lambda id: f" ({id + 1}-{self.residues[id]})"

        for residue_index in visited_residues_indices:
            path += format_(residue_index)
        return path

    def _write_pathways(self, output_file_path: str, header: str, pathways: List[Tuple[List[int], float]]) -> None:
        """
        Write generated pathways and their probabilities to an output file.

        The output file will include a header and a list of pathways, each with an index, its aggregated probability,
        and the formatted pathway string.

        Args:
            output_file_path (str): The path to the file where pathways will be saved.
            header (str): The header string to write at the beginning of the file.
            pathways (List[Tuple[List[int], float]]): A list of tuples, each containing a pathway and its probability.
        """
        with open(output_file_path, "w") as output:
            output.write(header + "\n")
            for index, pathway_and_probability in enumerate(pathways, start=1):
                pathway, probability = pathway_and_probability
                output.write(f"INDEX: {index}, PROBABILITY: {probability}, PATHWAY: {self._format_pathway(pathway)}\n")

    def _construct_pathways_output_path(self, output_directory: str, pathways_batch_index: int) -> str:
        """
        Construct the output file path for saving pathways.

        The file name is generated using the configuration's "pathways_file_name" format string and the provided batch index.

        Args:
            output_directory (str): The directory where the output file will be saved.
            pathways_batch_index (int): The batch index used in the file naming.

        Returns:
            str: The constructed output file path.
        """
        output_file_name: str = self.cls_config["pathways_file_name"].format(index=pathways_batch_index)
        return os.path.join(output_directory, output_file_name)

    ##################
    # PUBLIC METHODS #
    ##################

    def create_pathways(
        self, 
        start_residue: int,
        number_pathways: int,
        in_parallel: bool,
        target_residues: Optional[list[int]] = list(),
        number_steps: Optional[int] = None,
        pathways_batch_index: Optional[int] = 1,
        output_directory: Optional[str] = None, 
        num_workers: Optional[int] = None, 
        seed: Optional[int] = None
    ) -> str:
        if start_residue not in range(1, self._residues_range):
            raise ValueError(f"Invalid 'start_residue' argument. Expected an integer value in [0, {self._residues_range}]; Instead got: {start_residue}")
        
        if not isinstance(number_pathways, int) or number_pathways <= 0:
            raise ValueError(f"Invalid 'number_pathways' argument. Expected a positive integer value; Instead got: {number_pathways}")

        if number_steps is None:
            number_steps = self._residues_range
        if not isinstance(number_steps, int) or number_steps <= 0:
            raise ValueError(f"Invalid 'number_steps' argument. Expected a positive integer value; Instead got: {number_steps}")

        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        np.random.seed(seed)

        header: str = self.cls_config["pathways_file_header"].format(
            start_residue=start_residue,
            number_steps=number_steps,
            target_residues=target_residues,
            number_pathways=number_pathways,
            seed=seed
        )

        # needed due to 0-based indexing under the hood
        zero_indexed_start_residue = start_residue - 1
        zero_indexed_target_residues = tuple([i-1 for i in target_residues])

        if in_parallel:
            if num_workers is None:
                raise ValueError("If in_parallel=True, num_workers parameter must be provided")
            
            # Calculate batch sizes for parallel processing.
            pathway_batch_size, residual_pathways = divmod(number_pathways, num_workers)
            pathway_batches: List[int] = [pathway_batch_size + 1 if i < residual_pathways else pathway_batch_size for i in range(num_workers)]

            generated_pathways = _util.process_elementwise(
                in_parallel=True,
                Executor=ProcessPoolExecutor,
                max_workers=num_workers,
                capture_output=True
            )(
                pathway_batches,
                self._generate_multiple_pathways,
                zero_indexed_start_residue,
                number_steps,
                zero_indexed_target_residues
            )
            
        else:
            # For sequential processing, process all pathways as a single batch.
            pathway_batches: List[int] = [number_pathways]
            generated_pathways = _util.process_elementwise(
                in_parallel=False,
                capture_output=True
            )(
                pathway_batches,
                self._generate_multiple_pathways,
                zero_indexed_start_residue,
                number_steps,
                zero_indexed_target_residues
            )

        # Flatten the list of lists of pathways and sort them by aggregated probability (in descending order).
        generated_pathways = list(chain.from_iterable(generated_pathways))
        generated_pathways.sort(key=lambda x: x[1], reverse=True)

        output_file_path: str = self._construct_pathways_output_path(
            output_directory if output_directory is not None else os.getcwd(),
            pathways_batch_index
        )

        self._write_pathways(output_file_path, header, generated_pathways)

        return output_file_path

    def memory_cleanup(self) -> None:
        """
        Clean up the shared memory resources used by the Protein instance.

        This method closes and unlinks the shared memory blocks for both interaction and probability matrices,
        ensuring that resources are freed when they are no longer needed.
        """
        if self._memory_cleaned_up:
            return
        # Clean up shared memory blocks to free resources.
        self._interaction_matrices_shared_memory.close()
        self._interaction_matrices_shared_memory.unlink()
        self._probability_matrices_shared_memory.close()
        self._probability_matrices_shared_memory.unlink()
        self._memory_cleaned_up = True


if __name__ == "__main__":
    pass
