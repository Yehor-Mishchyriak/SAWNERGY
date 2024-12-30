#!AllostericPathwayAnalyzer/venv/bin/python3

# external imports
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import Union, Tuple

# local imports
import core
from .util import normalize_rows, frames_from_name, process_elementwise, init_error_handler_n_logger, generic_error_handler_n_logger


class ToMatricesConverter:
    """
    A class for converting residue interaction energy data from CSV files into interaction matrices 
    and transition probability matrices, and saving the results.

    Attributes:
        output_directory (str): Path to the directory where the output matrices will be saved.
    """

    @init_error_handler_n_logger(core.network_construction_logger)
    def __init__(self, output_directory_path: Union[str, None] = None) -> None:
        """
        Initializes the ToMatricesConverter class.

        Args:
            output_directory_path (str, optional): Path to the output directory. If None, a directory is created automatically.
        """
        self.output_directory = output_directory_path if output_directory_path else core.create_output_dir(
                                                        core.root_config["GLOBAL"]["output_directory_path"],
                                                        core.root_config["ToMatricesConverter"]["output_directory_name"])

        core.network_construction_logger.info(f"Successfully created the output directory for {self.__class__.__name__} class.")

    @generic_error_handler_n_logger(core.network_construction_logger)
    def _aggregate_energies(self, csv_file_path: str) -> pd.DataFrame:
        """
        Aggregates interatomic interaction energies for unique residue pairs from a CSV file,
        identified by their indices (e.g., residue_i_index and residue_j_index).
        For each unique pair, interaction energies are summed, and self-interactions are explicitly set to zero.

        Args:
            csv_file_path (str): Path to the input CSV file containing residue interaction data.

        Returns:
            pd.DataFrame: A DataFrame with aggregated energies for each residue pair.
        """
        core.network_construction_logger.info(f"Aggregating energies from file: {csv_file_path}")

        df = pd.read_csv(csv_file_path)
        df["pair"] = df.apply(lambda row: str((row["residue_i_index"], row["residue_j_index"])), axis=1) # axis=1 is rows; axis=0 is cols
        df["energy"] = df.apply(lambda row: 0.0 if row["residue_i_index"] == row["residue_j_index"] else row["energy"], axis=1) # set self-interactions to 0.0
        # group all the rows that share the "pair" column values into a collection and sum up the energy values
        aggregated_energies_by_pairs = df.groupby("pair")["energy"].sum().reset_index()

        core.network_construction_logger.info(f"Successfully aggregated energies from file: {csv_file_path}")
        return aggregated_energies_by_pairs

    @generic_error_handler_n_logger(core.network_construction_logger)
    def _to_interactions_probablities_matrices(aggregated_energies: pd.DataFrame, dimension: int) -> np.ndarray:
        """
        Converts aggregated interaction energies into interaction and transition probability matrices.
        Transition probability matrices are attained through row-vector-wise application of a normalisation function
        from util. For example, softmax or L2-norm.

        Args:
            aggregated_energies (pd.DataFrame): DataFrame with aggregated energies for each residue pair.
            dimension (int): Dimension of the resulting matrices.

        Returns:
            tuple: A tuple containing the interaction matrix and the transition probability matrix.
        """
        core.network_construction_logger.info("Converting aggregated energies to matrices.")

        pairs = aggregated_energies['pair'].apply(eval).tolist()
        energies = aggregated_energies['energy'].values
        matrix = np.zeros((dimension, dimension))

        for (i, j), energy in zip(pairs, energies):
            matrix[i, j] = energy
            matrix[j, i] = energy

        interactions_matrix = matrix[1:, 1:] # due to 1-based indexing of the residues
        probabilities_matrix = normalize_rows(interactions_matrix)

        core.network_construction_logger.info("Successfully converted aggregated energies to matrices.")
        return interactions_matrix, probabilities_matrix

    @generic_error_handler_n_logger(core.network_construction_logger)
    def _aggregate_convert_save(self, csv_file_path: str, dimension: int) -> None:
        """
        Aggregates energies, converts to matrices, and saves the resulting files.

        Args:
            csv_file_path (str): Path to the input CSV file.
            dimension (int): Dimension of the resulting matrices.
        """
        core.network_construction_logger.info(f"Processing file: {csv_file_path}")

        # AGGREGATE
        aggregated_energies = self._aggregate_energies(csv_file_path)
        # CONVERT
        interactions_matrix, probaility_matrix = self._to_interactions_probablities_matrices(aggregated_energies, dimension)
        # SAVE
            # get the csv file name
        csv_file_name = os.path.basename(csv_file_path)
            # extract the frames range for the file
        start_frame, end_frame = frames_from_name(csv_file_name)
            # construct the path for the container directory
        matrices_directory_path = os.path.join(self.output_directory,
                                        core.root_config["ToMatricesConverter"]["start_end_frames_dependent_matrices_directory_name"].format(start_frame, end_frame))
            # create the container directory
        os.makedirs(matrices_directory_path, exist_ok=True)
            # construct the paths for the output matrices
                # interactions
        interactions_output_path = os.path.join(matrices_directory_path, core.root_config["ToMatricesConverter"]["interactions_matrix_name"])
                # probabilities
        probabilities_output_path = os.path.join(matrices_directory_path, core.root_config["ToMatricesConverter"]["transition_probabilities_matrix_name"])
            # save the output
        np.save(interactions_output_path, interactions_matrix)
        np.save(probabilities_output_path, probaility_matrix)

        core.network_construction_logger.info(f"Successfully processed and saved matrices for file: {csv_file_path}")

    @generic_error_handler_n_logger(core.network_construction_logger)
    def process_target_directory(self, target_directory_path: str, dimension: int) -> str:
        """
        Processes all CSV files in the target directory, converting them to matrices and saving the results.

        Args:
            target_directory_path (str): Path to the directory containing input CSV files.
            dimension (int): Dimension of the resulting matrices.

        Returns:
            str: Path to the output directory.
        """
        # assumes the directory contains only the relevant .csv files
        csv_files_paths = (os.path.join(target_directory_path, file) for file in os.listdir(target_directory_path))

        if __name__ == "__main__":
            core.network_construction_logger.info(f"Began processing interatomic interaction energy csv files in parallel.")
            process_elementwise(in_parallel=True, Executor=ProcessPoolExecutor)(csv_files_paths, self._aggregate_convert_save, dimension)
        else:
            core.network_construction_logger.info(f"Began processing interatomic interaction energy csv files sequentially.")
            process_elementwise(in_parallel=False)(csv_files_paths, self._aggregate_convert_save, dimension)

        core.network_construction_logger.info(f"Successfully processed all the interatomic interaction energy csv files.")
        return self.output_directory

    @staticmethod
    @generic_error_handler_n_logger(core.network_construction_logger)
    def map_id_to_res(df: pd.DataFrame) -> Tuple[int, str]:
        """
        Maps residue IDs to residue names based on a DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame with residue IDs and names.

        Returns:
            tuple: A tuple mapping residue IDs to residue names.
        """
        core.network_construction_logger.info("Mapping residue IDs to names.")
        result = tuple(df.sort_values(by=["residue_i_index"]).drop_duplicates(subset=["residue_i_index"])["residue_i"])
        core.network_construction_logger.info("Successfully mapped residue IDs to names.")
        return result
    

def main():
    """
    Main function to execute the converter.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Target directory with optional output directory")
    parser.add_argument('target_directory', type=str, help='The directory containing the csv interaction energy files')
    parser.add_argument('--output_directory', type=str, default=None, help='The directory to save the matrices to')

    args = parser.parse_args()

    to_matrices_converter = ToMatricesConverter(args.output_directory)

    # return the output directory to the stdout
    print(to_matrices_converter.process_target_directory(args.target_directory))


if __name__ == "__main__":
    main()
