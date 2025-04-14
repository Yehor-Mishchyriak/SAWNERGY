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

    _defaults_to_residue_lvl = []

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

    # INTERACTIONS
    def _construct_int_prob_matrix_file_paths(self, csv_file_path: str, output_directory: str) -> Tuple[str, str]:

        csv_file_name, analysis_type = _util.name_and_analysis_type_from_path(csv_file_path)
        if analysis_type == "com":
            raise ValueError(f"Expected a csv file with interactions data; instead got: {analysis_type}")

        start_frame, end_frame = _util.frames_from_name(csv_file_name)
        container_dir_name = self.cls_config["matrices_directory_name"].format(start_frame=start_frame, end_frame=end_frame)
        matrices_directory_path = os.path.join(output_directory, analysis_type, container_dir_name)

        # construct the paths for the future interactions and probabilities matrices 
        interactions_output_path = os.path.join(_util.new_dir_at(matrices_directory_path), self.cls_config["interactions_matrix_name"])
        probabilities_output_path = os.path.join(_util.new_dir_at(matrices_directory_path), self.cls_config["probabilities_matrix_name"])

        return interactions_output_path, probabilities_output_path
    
    # aggregate
    @staticmethod
    def residue_int_from_atomic_int(csv_file_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_file_path)
        _, analysis_type = _util.name_and_analysis_type_from_path(csv_file_path)
        if analysis_type in ToMatricesConverter._defaults_to_residue_lvl:
            return df
        # group by the residue index columns, summing the energy values
        aggregated_energies = df.groupby(["residue_i_index", "residue_j_index"], as_index=False)["energy"].sum()
        return aggregated_energies
    
    # convert
    @staticmethod
    def to_interactions_probablities_matrices(energies: pd.DataFrame, normalisation_function = None, null_prob_value = None) -> Tuple[np.ndarray, np.ndarray]:
        
        if normalisation_function is None:
            normalisation_function = _util.l1
        if null_prob_value is None:
            null_prob_value = 0.0
        
        # extract indices and energy values
        rows = energies["residue_i_index"].to_numpy(dtype=np.intp)
        cols = energies["residue_j_index"].to_numpy(dtype=np.intp)
        energies = energies["energy"].to_numpy(dtype=np.float64)
        
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
        soon_probabilities_matrix = matrix.copy()
        np.fill_diagonal(soon_probabilities_matrix, null_prob_value) # in case of softmax, we'd use -np.inf, for l_n norm, we use 0.0

        # normalize the probabilities matrix
        probabilities_matrix = _util.normalize_rows(soon_probabilities_matrix, normalisation_function)
        # set the diagonal to the smallest possible value to allow safe logarithm later on
        np.fill_diagonal(probabilities_matrix, float_info.min)
        
        # create interactions matrix: copy and set self-interactions to zero
        interactions_matrix = matrix.copy()
        np.fill_diagonal(interactions_matrix, 0.0)
        
        return interactions_matrix, probabilities_matrix

    def aggregate_convert_save(self, csv_file_path: str, output_directory: str, normalisation_function = None, null_prob_value = None) -> None:
        res_lvl_energies = ToMatricesConverter.residue_int_from_atomic_int(csv_file_path)
        interactions_matrix, probaility_matrix = ToMatricesConverter.to_interactions_probablities_matrices(res_lvl_energies, normalisation_function, null_prob_value)

        interactions_output_path, probabilities_output_path = self._construct_int_prob_matrix_file_paths(csv_file_path, output_directory)
        np.save(interactions_output_path, interactions_matrix)
        np.save(probabilities_output_path, probaility_matrix)

    def process_multiple_csv_files(self, csv_files_to_process: List[str], output_directory: str, normalisation_function = None, null_prob_value = None) -> None:
        for file in csv_files_to_process:
            self.aggregate_convert_save(file, output_directory, normalisation_function, null_prob_value)

    # COORDINATES
    def coords_to_np(self, com_csv_dir_path: str, output_directory: str):
        sorted_by_resid_com_files = sorted(os.listdir(com_csv_dir_path), key=_util.residue_id_from_name)
        csv_file_paths = [os.path.join(com_csv_dir_path, com_file) for com_file in sorted_by_resid_com_files]

        residue_coords = []
        for csv_file_path in csv_file_paths:
            data = np.loadtxt(csv_file_path, delimiter=',', skiprows=1)[1:4]
            residue_coords.append(data)
                                                                               #    0,          1,            2
        residue_coords = np.array(residue_coords) # 3D NumPy array of shape = (# residues, # frames, # coordinate_axes)
                                                                                           #    1,          0,            2
        frames = np.transpose(residue_coords, axes=(1, 0, 2)) # 3D NumPy array of shape = (# frames, # residues, # coordinate_axes)

        output_directory = _util.new_dir_at(os.path.join(output_directory, self.cls_config["coordinates_directory_name"]))
        matrix_path_template = os.path.join(output_directory, self.cls_config["coordinates_file_name_template"])

        for frame_id in range(frames.shape[0]):
            frame = frames[frame_id]
            np.save(matrix_path_template.format(frame_id=frame_id), frame)

    def create_id_to_res_map():
        pass

    def process_target_directory(self, target_directory_path: str,
                                 in_parallel: bool, allowed_memory_percentage_hint: Optional[float] = None,
                                 num_workers: Optional[int] = None, output_directory_path: Optional[str] = None,
                                 normalisation_function = None, null_prob_value = None) -> str:
        output_directory = output_directory_path if output_directory_path else _util.create_output_dir(os.getcwd(), self.cls_config["output_directory_name_template"])

        if in_parallel:
            if num_workers is None:
                raise ValueError("If in_parallel=True, num_workers parameter must be provided")
            if allowed_memory_percentage_hint is None:
                raise ValueError("If in_parallel=True, allowed_memory_percentage_hint parameter must be provided")
            convert_to_matrices = _util.process_elementwise(in_parallel=True, Executor=ProcessPoolExecutor, capture_output=False, max_workers=num_workers)
        else:
            convert_to_matrices = _util.process_elementwise(in_parallel=False, capture_output=False)
        
        target_subdirectory_paths = [os.path.join(target_directory_path, target_subdirectory) for target_subdirectory in os.listdir(target_directory_path)]
        if in_parallel:
            for target_subdirectory_path in target_subdirectory_paths:
                if os.path.basename(target_directory_path) == "com":
                    self.coords_to_np(target_directory_path, output_directory)
                convert_to_matrices(_util.chunked_dir(target_subdirectory_path, allowed_memory_percentage_hint, num_workers),
                                    self.process_multiple_csv_files, output_directory, normalisation_function, null_prob_value)
        else:
            for target_subdirectory_path in target_subdirectory_paths:
                if os.path.basename(target_directory_path) == "com":
                    self.coords_to_np(target_directory_path, output_directory)
                convert_to_matrices([os.path.join(target_subdirectory_path, file) for file in os.listdir(target_subdirectory_path)],
                                    self.aggregate_convert_save, output_directory, normalisation_function, null_prob_value)

        self.create_id_to_res_map()

        return output_directory
    

if __name__ == "__main__":
    pass
