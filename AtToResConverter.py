#!/usr/bin/env python3

import os
from datetime import datetime
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AtToResConverter:
    """
    A class to convert the output of pairwise command of cpptraj to numpy interaction matrices.

    The output data is of the following format:
    - "Residue_Level_<time_when_created>" dir that contains "aggregated_energies" and "interaction_matrices" dirs, which contain
    - "residue_level_<i>-<j>.csv" and "interactions_matrix_residue_level_<i>-<j>.npy" files,
    - where i is the start frame and j is the end frame indices.

    Attributes:
        target_directory (str): The directory where input files are stored.
        save_output (bool): Flag to save the output files.
        output_directory (str): Directory to save output files, if save_output is True.
        aggregated_energies_directory (str): Directory to save aggregated energies.
        interaction_matrices_directory (str): Directory to save interaction matrices.
    """

    def __init__(self, target_directory: str = None, save_output: bool = True, output_directory=None) -> None:
        """
        Initialize the AtToResConverter class.

        Args:
            target_directory (str): The target directory for input files.
            save_output (bool): Flag to save the output files.
            output_directory (str, optional): Directory to save output files, if save_output is True.
        """
        self.target_directory = target_directory if target_directory else os.getcwd()
        self.save_output = save_output

        if self.save_output:
            output_directories = self._create_output_dir(output_directory)
            self.output_directory, self.aggregated_energies_directory, self.interaction_matrices_directory = output_directories
        else:
            self.output_directory = os.getcwd()

    @staticmethod
    def _create_output_dir(output_directory: str = None) -> tuple:
        """
        Create directories for storing output files.

        Args:
            output_directory (str, optional): The base directory to create the output directories in.

        Returns:
            tuple: Paths of the output directories.

        Raises:
            OSError: If there is an error creating the directories.
        """
        try:
            current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
            root_directory = output_directory if output_directory else os.getcwd()
            output_directory = os.path.join(root_directory, f"Residue_Level_{current_time}")
            aggregated_energies_directory = os.path.join(output_directory, "aggregated_energies")
            interaction_matrices_directory = os.path.join(output_directory, "interaction_matrices")

            os.makedirs(aggregated_energies_directory, exist_ok=True)
            os.makedirs(interaction_matrices_directory, exist_ok=True)

            logging.info(f"Created output directories: {output_directory}, {aggregated_energies_directory}, {interaction_matrices_directory}")

            return output_directory, aggregated_energies_directory, interaction_matrices_directory
        except OSError as e:
            logging.error(f"Error creating output directories: {e}")
            raise

    def aggregate_energies(self, input_file: str) -> pd.DataFrame:
        """
        Aggregate interaction energies by residue pairs from a CSV file.

        The input CSV file must have the following schema:
        - residue_i_index (int): Index of the first residue.
        - residue_j_index (int): Index of the second residue.
        - energy (float): Interaction energy between the residues' atoms.

        Args:
            input_file (str): Path to the input CSV file.

        Returns:
            pd.DataFrame: Aggregated interaction energies.

        Raises:
            FileNotFoundError: If the input file does not exist.
            pd.errors.EmptyDataError: If the input file is empty.
            Exception: If there is an error reading or processing the file.
        """
        try:
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file {input_file} does not exist.")

            df = pd.read_csv(input_file)
            
            df["pair"] = df.apply(lambda row: str((row["residue_i_index"], row["residue_j_index"])), axis=1)
            df["energy"] = df.apply(lambda row: 0.0 if row["residue_i_index"] == row["residue_j_index"] else row["energy"], axis=1)
            aggregated = df.groupby("pair")["energy"].sum().reset_index()

            if self.save_output:
                output_file = f"residue_level_{os.path.basename(input_file)}"
                output_path = os.path.join(self.aggregated_energies_directory, output_file)
                aggregated.to_csv(output_path, index=False)
                logging.info(f"Aggregated energies saved to {output_path}")

            return aggregated
        except FileNotFoundError as fnf_error:
            logging.error(fnf_error)
            raise
        except pd.errors.EmptyDataError as ede_error:
            logging.error(f"Input file {input_file} is empty: {ede_error}")
            raise
        except Exception as e:
            logging.error(f"Error aggregating energies from file {input_file}: {e}")
            raise
    
    def convert_to_numpy(self, aggregate_energies: pd.DataFrame, output_file_name: str = None) -> np.ndarray:
        """
        Convert aggregated energies DataFrame to a numpy array interaction matrix.

        The aggregated DataFrame must have the following schema:
        - pair (object): indices of the pair of residues.
        - energy (float): Interaction energy between the residues.

        Args:
            aggregate_energies (pd.DataFrame): Aggregated interaction energies.
            output_file_name (str, optional): Optional file name to save the numpy array.

        Returns:
            np.ndarray: Interaction matrix as a numpy array.

        Raises:
            ValueError: If there is an error during conversion.
            Exception: If there is a general error during conversion.
        """
        try:
            pairs = aggregate_energies['pair'].apply(eval).tolist()
            energies = aggregate_energies['energy'].values
            max_index = max(max(pair) for pair in pairs) + 1
            matrix = np.zeros((max_index, max_index))

            for (i, j), energy in zip(pairs, energies):
                matrix[i, j] = energy
                matrix[j, i] = energy

            matrix = matrix[1:, 1:]

            if self.save_output:
                if not output_file_name:
                    raise ValueError("Output file name must be provided when saving output.")
                np.save(os.path.join(self.interaction_matrices_directory, output_file_name), matrix)
                logging.info(f"Interaction matrix saved to {output_file_name}")

            return matrix

        except ValueError as ve:
            logging.error(ve)
            raise
        except Exception as e:
            logging.error(f"Error converting DataFrame to numpy array: {e}")
            raise

    def _sequential_processor(self) -> list:
        """
        Process all CSV files in the target directory sequentially.

        Returns:
            list: List of interaction matrices as numpy arrays.

        Raises:
            Exception: If there is an error processing the files sequentially.
        """
        try:
            interaction_matrices = []
            for analysis_file in os.listdir(self.target_directory):
                if analysis_file.endswith(".csv"):
                    full_path = os.path.join(self.target_directory, analysis_file)
                    aggregated_energies = self.aggregate_energies(full_path)
                    output_file_name = f"interactions_matrix_{analysis_file.replace('.csv', '.npy')}"
                    interaction_matrices.append(self.convert_to_numpy(aggregated_energies, output_file_name=output_file_name))
            logging.info(f"Processed all .csv files sequentially in directory: {self.target_directory}")
            return interaction_matrices
        except Exception as e:
            logging.error(f"Error processing files sequentially: {e}")
            raise

    def _parallel_processor(self) -> list:
        """
        Process all CSV files in the target directory in parallel.

        Returns:
            list: List of interaction matrices as numpy arrays.

        Raises:
            Exception: If there is an error processing the files in parallel.
        """
        try:
            aggregated_energies_futures = []
            interaction_matrices = []

            with ProcessPoolExecutor() as executor:
                for analysis_file in os.listdir(self.target_directory):
                    if analysis_file.endswith(".csv"):
                        full_path = os.path.join(self.target_directory, analysis_file)
                        aggregated_energies_futures.append((executor.submit(self.aggregate_energies, full_path), analysis_file))

                logging.info(f"Submitted tasks for aggregating energies in directory: {self.target_directory}")

                for future, analysis_file in aggregated_energies_futures:
                    try:
                        aggregated_energies = future.result()
                        output_file_name = f"interactions_matrix_{analysis_file.replace('.csv', '.npy')}"
                        interaction_matrices.append(executor.submit(self.convert_to_numpy, aggregated_energies, output_file_name=output_file_name))
                    except Exception as e:
                        logging.error(f"Error processing aggregated energies for file {analysis_file}: {e}")
                        raise

            final_matrices = []
            for future in as_completed(interaction_matrices):
                try:
                    final_matrices.append(future.result())
                except Exception as e:
                    logging.error(f"Error converting to numpy: {e}")
                    raise

            return final_matrices
        except Exception as e:
            logging.error(f"Error processing files in parallel: {e}")
            raise

    def process_target_directory(self) -> list:
        """
        Process all CSV files in the target directory.

        Returns:
            list: List of interaction matrices as numpy arrays.

        Raises:
            Exception: If there is an error processing files in the directory.
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

    parser = argparse.ArgumentParser(description="Convert cpptraj pairwise command output to numpy interaction matrices.")
    parser.add_argument('target_directory', type=str, help='The directory containing the CSV analysis files.')
    parser.add_argument('--output_directory', type=str, default=None, help='The directory to save the output files to.')

    args = parser.parse_args()

    try:
        at_to_res_converter = AtToResConverter(
            target_directory=args.target_directory,
            save_output=True,
            output_directory=args.output_directory
        )
        at_to_res_converter.process_target_directory()
        print(at_to_res_converter.output_directory)
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
