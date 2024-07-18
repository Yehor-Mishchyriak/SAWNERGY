import os
from datetime import datetime
import numpy as np
from interfaces.InteractionsToProbabilitiesABC import InteractionsToProbabilitiesABC
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import util

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ElectrostaticInteractionsToProbabilities(InteractionsToProbabilitiesABC):
    """
    A class to process interaction matrices, convert them to probability matrices, and save them to an output directory.

    Attributes:
        target_directory (str): The directory containing the interaction matrix files.
        output_directory (str): The directory to save the output probability matrix files.
    """

    def __init__(self, target_directory: str = None, output_directory: str = None) -> None:
        """
        Initialize the ElectrostaticInteractionsToProbabilities with the target and output directories.

        Args:
            target_directory (str, optional): The directory containing the interaction matrix files.
            output_directory (str, optional): The directory to save the output probability matrix files.
        """
        self.target_directory = target_directory if target_directory else os.getcwd()
        self.output_directory = self._create_output_dir(output_directory)

    @staticmethod
    def _create_output_dir(output_directory: str = None) -> str:
        """
        Create a unique output directory based on the current time.

        Args:
            output_directory (str, optional): The base directory to create the output directory in.

        Returns:
            str: The path to the created output directory.

        Raises:
            OSError: If there is an error creating the directory.
        """
        try:
            current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
            root_directory = output_directory if output_directory else os.getcwd()
            output_directory = os.path.join(root_directory, f"OUTPUT_{current_time}")
            os.makedirs(output_directory, exist_ok=True)
            probability_matrices_directory = os.path.join(output_directory, "probability_matrices")
            os.makedirs(probability_matrices_directory, exist_ok=True)
            logging.info(f"Created output directory: {probability_matrices_directory}")
            return probability_matrices_directory
        except OSError as e:
            logging.error(f"Error creating output directory: {e}")
            raise

    @staticmethod
    def convert_to_probabilities(interactions_matrix: np.array) -> np.array:
        """
        Convert an interaction matrix to a probability matrix using a utility function.

        Args:
            interactions_matrix (np.array): The interaction matrix to convert.

        Returns:
            np.array: The converted probability matrix.
        """
        return util.transition_probs_from_interactions(interactions_matrix)

    def __sequential_processor(self) -> list:
        """
        Process the target directory sequentially to convert interaction matrices to probability matrices.

        Returns:
            list: List of converted probability matrices.
        """
        try:
            probability_matrices = []
            for npy_file in os.listdir(self.target_directory):
                if npy_file.endswith(".npy"):
                    path = os.path.join(self.target_directory, npy_file)
                    interaction_matrix = np.load(path)
                    probability_matrix = self.convert_to_probabilities(interaction_matrix)
                    probability_matrices.append(probability_matrix)
                    output_file_name = f"probabilities_from_{npy_file.replace('.npy', '')}.npy"
                    np.save(os.path.join(self.output_directory, output_file_name), probability_matrix)
            logging.info(f"Sequential processing complete. Processed {len(probability_matrices)} files.")
            return probability_matrices
        except Exception as e:
            logging.error(f"Error in sequential processing: {e}")
            raise

    def __parallel_processor(self) -> list:
        """
        Process the target directory in parallel to convert interaction matrices to probability matrices.

        Returns:
            list: List of converted probability matrices.
        """
        probability_matrices_futures = []
        with ProcessPoolExecutor() as executor:
            for npy_file in os.listdir(self.target_directory):
                if npy_file.endswith(".npy"):
                    path = os.path.join(self.target_directory, npy_file)
                    interaction_matrix = np.load(path)
                    future = executor.submit(self.convert_to_probabilities, interaction_matrix)
                    probability_matrices_futures.append((future, npy_file))
            logging.info("Parallel processing initiated.")

        probability_matrices = []
        for future, npy_file in as_completed(probability_matrices_futures):
            try:
                probability_matrix = future.result()
                probability_matrices.append(probability_matrix)
                output_file_name = f"probabilities_from_{npy_file.replace('.npy', '')}.npy"
                np.save(os.path.join(self.output_directory, output_file_name), probability_matrix)
            except Exception as e:
                logging.error(f"Error processing file {npy_file}: {e}")
                raise
        logging.info(f"Parallel processing complete. Processed {len(probability_matrices)} files.")
        return probability_matrices

    def process_target_directory(self) -> list:
        """
        Process the target directory of interactions matrices.

        Returns:
            list: List of converted probability matrices.
        """
        if __name__ == "__main__":
            result = self.__parallel_processor()
        else:
            result = self.__sequential_processor()
        return result


def main():
    """
    Main function to execute the processor.
    """
    pass


if __name__ == "__main__":
    main()
