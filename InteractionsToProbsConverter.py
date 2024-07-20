#!/usr/bin/env python3

import os
from shutil import copy
from datetime import datetime
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import util

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InteractionsToProbsConverter:
    """
    A class to process interaction matrices, convert them to probability matrices, and save them to an output directory.

    Attributes:
        target_directory (str): The directory containing the interaction matrices.
        output_directory (str): The directory to save the output pairs of interaction and probability matrices.
    """

    def __init__(self, target_directory: str = None, save_output: bool = True, output_directory: str = None) -> None:
        """
        Initialize the InteractionsToProbsConverter with the target and output directories.

        The output data is of the following format:
        - "Paired_Probs_Energies_<time_when_created>" dir that contains "<i>-<j>" dirs;
        - each "<i>-<j>" dir contains two files: probabilities_matrix_residue_level_(<i>-<j>).npy and interactions_matrix_residue_level_(<i>-<j>).npy,
        - where i is the start frame and j is the end frame indices;

        Args:
            target_directory (str, optional): The directory containing the interaction matrix files.
            output_directory (str, optional): The directory to save the output probability matrix files.
        """
        self.target_directory = target_directory if target_directory else os.getcwd()
        self.save_output = save_output
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
            output_directory = os.path.join(root_directory, f"Paired_Probs_Energies_{current_time}")
            os.makedirs(output_directory, exist_ok=True)
            logging.info(f"Created output directory: {output_directory}")
            return output_directory
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
        try:
            return util.transition_probs_from_interactions(interactions_matrix)
        except Exception as e:
            logging.error(f"Error converting interactions matrix to probabilities: {e}")
            raise

    def _create_container_dir(self, container_dir_name: str) -> str:
        """
        Create a directory for storing output files.

        Args:
            container_dir_name (str): The name of the container directory.

        Returns:
            str: The path to the created container directory.
        """
        try:
            container_dir = os.path.join(self.output_directory, container_dir_name)
            os.makedirs(container_dir, exist_ok=True)
            logging.info(f"Created container directory: {container_dir}")
            return container_dir
        except OSError as e:
            logging.error(f"Error creating container directory {container_dir_name}: {e}")
            raise

    def _sequential_processor(self) -> list:
        """
        Process the target directory sequentially to convert interaction matrices to probability matrices.

        Returns:
            list: List of converted probability matrices.
        """
        try:
            probability_matrices = []
            for npy_file in os.listdir(self.target_directory):
                if npy_file.endswith(".npy"):
                    path_to_npy_file = os.path.join(self.target_directory, npy_file)
                    interaction_matrix = np.load(path_to_npy_file)
                    probability_matrix = self.convert_to_probabilities(interaction_matrix)
                    probability_matrices.append(probability_matrix)

                    if self.save_output:
                        original_frames_range = util.extract_frames_range(npy_file)
                        start_frame, end_frame = original_frames_range
                        output_directory_name = f"{start_frame}-{end_frame}"
                        output_file_name = npy_file.replace("interactions", "probabilities")
                        save_to = self._create_container_dir(output_directory_name)
                        # Save probabilities matrix
                        np.save(os.path.join(save_to, output_file_name), probability_matrix)
                        # Copy interactions matrix
                        copy(path_to_npy_file, os.path.join(save_to, npy_file))
                        logging.info(f"Processed and saved file: {npy_file}")

            logging.info(f"Sequential processing complete. Processed {len(probability_matrices)} files.")
            return probability_matrices

        except Exception as e:
            logging.error(f"Error in sequential processing: {e}")
            raise

    def _parallel_processor(self) -> list:
        """
        Process the target directory in parallel to convert interaction matrices to probability matrices.

        Returns:
            list: List of converted probability matrices.
        """
        probability_matrices_futures = []
        try:
            with ProcessPoolExecutor() as executor:
                for npy_file in os.listdir(self.target_directory):
                    if npy_file.endswith(".npy"):
                        path_to_npy_file = os.path.join(self.target_directory, npy_file)
                        future = executor.submit(self.convert_to_probabilities, np.load(path_to_npy_file))
                        probability_matrices_futures.append((future, npy_file))
                logging.info("Parallel processing initiated.")

            probability_matrices = []
            for future, npy_file in as_completed(probability_matrices_futures):
                try:
                    probability_matrix = future.result()
                    probability_matrices.append(probability_matrix)

                    if self.save_output:
                        original_frames_range = util.extract_frames_range(npy_file)
                        start_frame, end_frame = original_frames_range
                        output_directory_name = f"{start_frame}-{end_frame}"
                        output_file_name = npy_file.replace("interactions", "probabilities")
                        save_to = self._create_container_dir(output_directory_name)
                        # Save probabilities matrix
                        np.save(os.path.join(save_to, output_file_name), probability_matrix)
                        # Copy interactions matrix
                        copy(path_to_npy_file, os.path.join(save_to, npy_file))
                        logging.info(f"Processed and saved file: {npy_file}")

                except Exception as e:
                    logging.error(f"Error processing file {npy_file}: {e}")
                    raise

            logging.info(f"Parallel processing complete. Processed {len(probability_matrices)} files.")
            return probability_matrices

        except Exception as e:
            logging.error(f"Error during parallel processing: {e}")
            raise

    def process_target_directory(self) -> list:
        """
        Process the target directory of interactions matrices.

        Returns:
            list: List of converted probability matrices.
        """
        try:
            if __name__ == "__main__":
                return self._parallel_processor()
            else:
                return self._sequential_processor()
        except Exception as e:
            logging.error(f"Error processing target directory: {e}")
            raise


def main():
    """
    Main function to execute the processor.
    """

    import argparse

    parser = argparse.ArgumentParser(description="Process interaction matrices to probability matrices.")
    parser.add_argument('target_directory', type=str, help='The directory containing the interaction matrix files.')
    parser.add_argument('--output_directory', type=str, default=None, help='The directory to save the output probability matrix files.')

    args = parser.parse_args()

    try:
        interactions_to_probs_converter = InteractionsToProbsConverter(
            target_directory=args.target_directory,
            output_directory=args.output_directory
        )
        interactions_to_probs_converter.process_target_directory()
        print(interactions_to_probs_converter.output_directory)
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
