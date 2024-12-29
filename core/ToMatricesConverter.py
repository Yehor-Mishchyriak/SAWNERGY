#!AllostericPathwayAnalyzer/venv/bin/python3

# external imports
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# local imports
import core
from .util import normalize_rows, frames_from_name, process_elementwise


class ToMatricesConverter:

    def __init__(self, output_directory_path: str = None) -> None:

        self.output_directory = output_directory_path if output_directory_path else core.create_output_dir(
                                                        core.root_config["GLOBAL"]["output_directory_path"],
                                                        core.root_config["ToMatricesConverter"]["output_directory_name"])

        core.network_construction_logger.info(f"Successfully created the output directory for {self.__class__.__name__} class.")


    def _aggregate_energies(self, csv_file_path: str) -> pd.DataFrame:

        df = pd.read_csv(csv_file_path)
        df["pair"] = df.apply(lambda row: str((row["residue_i_index"], row["residue_j_index"])), axis=1) # axis=1 is rows; axis=0 is cols
        df["energy"] = df.apply(lambda row: 0.0 if row["residue_i_index"] == row["residue_j_index"] else row["energy"], axis=1) # set self-interactions to 0.0
        # group all the rows that share the "pair" column values into a collection and sum up the energy values
        aggregated_energies_by_pairs = df.groupby("pair")["energy"].sum().reset_index()

        return aggregated_energies_by_pairs

    def _to_interactions_probablities_matrices(aggregated_energies: pd.DataFrame, dimension: int) -> np.ndarray:
        pairs = aggregated_energies['pair'].apply(eval).tolist()
        energies = aggregated_energies['energy'].values
        matrix = np.zeros((dimension, dimension))

        for (i, j), energy in zip(pairs, energies):
            matrix[i, j] = energy
            matrix[j, i] = energy

        interactions_matrix = matrix[1:, 1:] # due to 1-based indexing of the residues
        probabilities_matrix = normalize_rows(interactions_matrix)

        return interactions_matrix, probabilities_matrix

    def _aggregate_convert_save(self, csv_file_path: str, dimension: int) -> None:
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

    def process_target_directory(self, target_directory_path: str, dimension: int) -> str:
        # assumes the directory contains only the relevant .csv files
        csv_files_paths = (os.path.join(target_directory_path, file) for file in os.listdir(target_directory_path))

        if __name__ == "__main__":
            process_elementwise(in_parallel=True, Executor=ProcessPoolExecutor)(csv_files_paths, self._aggregate_convert_save, dimension)
        else:
            process_elementwise(in_parallel=False)(csv_files_paths, self._aggregate_convert_save, dimension)

        return self.output_directory

    @staticmethod
    def map_id_to_res(df: pd.DataFrame):
        return tuple(df.sort_values(by=["residue_i_index"]).drop_duplicates(subset=["residue_i_index"])["residue_i"])


def main():
    """To be filled in"""
    pass


if __name__ == "__main__":
    main()
