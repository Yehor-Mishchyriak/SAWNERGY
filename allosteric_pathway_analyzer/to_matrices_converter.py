# external imports
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sys import float_info
from typing import Optional, List, Tuple, Any

# local imports
from . import pkg_globals
from . import _util

class ToMatricesConverter:
    def __init__(self, config: Optional[dict] = None) -> None:
        self.global_config = None
        self.cls_config = None
        if config is None:
            self.set_config(pkg_globals.default_config)
        else:
            self.set_config(config)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.cls_config})"

    def set_config(self, config: dict) -> None:
        self.global_config = config
        self.cls_config = config[self.__class__.__name__]

    def _construct_matrices_output_paths(self, csv_file_path: str, output_directory: str) -> Tuple[str, str]:
        # extract the frames range from the csv file name
        csv_file_name = os.path.basename(csv_file_path)
        start_frame, end_frame = _util.frames_from_name(csv_file_name)

        # construct the output directory path
        container_dir_name = self.cls_config["matrices_directory_name"].format(start=start_frame, end=end_frame)
        matrices_directory_path = os.path.join(output_directory, container_dir_name)

        # create the output directory
        os.makedirs(matrices_directory_path, exist_ok=True)

        # construct the paths for the future interactions and probabilities matrices 
        interactions_output_path = os.path.join(matrices_directory_path, self.cls_config["interactions_matrix_name"])
        probabilities_output_path = os.path.join(matrices_directory_path, self.cls_config["probabilities_matrix_name"])

        return interactions_output_path, probabilities_output_path

    @staticmethod
    def aggregate_energies(csv_file_path: str) -> pd.DataFrame:
        # read the CSV file into a df
        df = pd.read_csv(csv_file_path)
        
        # group by the residue index columns, summing the energy values
        aggregated_energies = df.groupby(["residue_i_index", "residue_j_index"], as_index=False)["energy"].sum()
        return aggregated_energies

    @staticmethod
    def to_interactions_probablities_matrices(aggregated_energies: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # extract indices and energy values
        rows = aggregated_energies["residue_i_index"].to_numpy(dtype=np.intp)
        cols = aggregated_energies["residue_j_index"].to_numpy(dtype=np.intp)
        energies = aggregated_energies["energy"].to_numpy(dtype=np.float64)
        
        # determine the dimension of the matrix
        dim = max(rows.max(), cols.max()) + 1
        matrix = np.zeros((dim, dim), dtype=np.float64)
        
        # populate the matrix with energy values
        matrix[rows, cols] = energies
        matrix[cols, rows] = energies  # ensure symmetry
        
        # discard the borders of the matrix due to 1-based indexing of residues
        matrix = matrix[1:, 1:]
        
        # convert all values to their absolute value
        matrix = np.abs(matrix)
        
        # create probabilities matrix: copy and set diagonal to -infinity for normalization
        probabilities_matrix = matrix.copy()
        np.fill_diagonal(probabilities_matrix, -np.inf)
        # normalize the probabilities matrix
        probabilities_matrix = _util.normalize_rows(probabilities_matrix)
        # set the diagonal to the smallest possible value to allow safe logarithm later on
        np.fill_diagonal(probabilities_matrix, float_info.min)
        
        # create interactions matrix: copy and set self-interactions to zero
        interactions_matrix = matrix.copy()
        np.fill_diagonal(interactions_matrix, 0.0)
        
        return interactions_matrix, probabilities_matrix

    def aggregate_convert_save(self, csv_file_path: str, output_directory: str) -> None:
        aggregated_energies = ToMatricesConverter.aggregate_energies(csv_file_path)
        interactions_matrix, probaility_matrix = ToMatricesConverter.to_interactions_probablities_matrices(aggregated_energies)

        interactions_output_path, probabilities_output_path = self._construct_matrices_output_paths(csv_file_path, output_directory)
        np.save(interactions_output_path, interactions_matrix)
        np.save(probabilities_output_path, probaility_matrix)

    def process_multiple_csv_files(self, csv_files_to_process: List[str], output_directory: str) -> None:
        for file in csv_files_to_process:
            self.aggregate_convert_save(file, output_directory)

    def process_target_directory(self, target_directory_path: str,
                                 in_parallel: bool, allowed_memory_percentage_hint: Optional[float] = None,
                                 num_workers: Optional[int] = None, output_directory_path: Optional[str] = None,
                                 create_id_to_res_map: Optional[bool] = True) -> str:
        output_directory = output_directory_path if output_directory_path else _util.create_output_dir(os.getcwd(), self.cls_config["output_directory_name"])
        csv_file_paths = [os.path.join(target_directory_path, file) for file in os.listdir(target_directory_path)]

        if in_parallel:
            if num_workers is None:
                raise ValueError("If in_parallel=True, num_workers parameter must be provided")
            if allowed_memory_percentage_hint is None:
                raise ValueError("If in_parallel=True, allowed_memory_percentage_hint parameter must be provided")
            
            _util.process_elementwise(
                in_parallel=True,
                Executor=ProcessPoolExecutor,
                capture_output=False,
                max_workers=num_workers
            )(
                _util.chunked_dir(target_directory_path, allowed_memory_percentage_hint, num_workers),
                self.process_multiple_csv_files,
                output_directory
            )
        else:
            _util.process_elementwise(
                in_parallel=False,
                capture_output=False,
            )(
                csv_file_paths,
                self.aggregate_convert_save,
                output_directory
            )

        if create_id_to_res_map:
            self.map_id_to_res(csv_file_paths[0], output_directory)

        return output_directory

    def map_id_to_res(self, csv_file_path: str, output_directory_path: Optional[str] = None) -> Tuple[Any, ...]:
        output_directory = output_directory_path if output_directory_path else _util.create_output_dir(os.getcwd(), self.cls_config["output_directory_name"])
        df = pd.read_csv(csv_file_path)
        result = tuple(df.sort_values(by=["residue_i_index"]).drop_duplicates(subset=["residue_i_index"])["residue_i"])
        output_file_path = os.path.join(output_directory, self.cls_config["id_to_res_map_name"])
        with open(output_file_path, "w") as output_file:
            output_file.write(str(result))

        return result
    

if __name__ == "__main__":
    pass
