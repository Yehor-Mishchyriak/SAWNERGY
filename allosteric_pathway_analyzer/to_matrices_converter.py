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
    def __init__(self) -> None:
        """
        Initialize the ToMatricesConverter with the default configuration.

        The configuration is set using the default configuration from pkg_globals,
        based on the class name.
        """
        self.config = None
        self.set_config(pkg_globals.default_config[self.__class__.__name__])

    def __repr__(self) -> str:
        """
        Return a string representation of the ToMatricesConverter instance.
        
        Returns:
            str: A string that represents the instance including its configuration.
        """
        return f"{self.__class__.__name__}(config={self.config})"

    def set_config(self, config: dict) -> None:
        """
        Set the configuration for the converter.

        This method validates that all required configuration keys are present.
        
        Args:
            config (dict): Configuration dictionary containing the following keys:
                - "output_directory_name"
                - "matrices_directory_name"
                - "interactions_matrix_name"
                - "probabilities_matrix_name"
                - "id_to_res_map_name"
        
        Raises:
            ValueError: If any required configuration key is missing.
        """
        if "output_directory_name" not in config:
            raise ValueError(f"Invalid {self.__class__.__name__} config: missing output_directory_name field")
        if "matrices_directory_name" not in config:
            raise ValueError(f"Invalid {self.__class__.__name__} config: missing matrices_directory_name field")
        if "interactions_matrix_name" not in config:
            raise ValueError(f"Invalid {self.__class__.__name__} config: missing interactions_matrix_name field")
        if "probabilities_matrix_name" not in config:
            raise ValueError(f"Invalid {self.__class__.__name__} config: missing probabilities_matrix_name field")
        if "id_to_res_map_name" not in config:
            raise ValueError(f"Invalid {self.__class__.__name__} config: missing id_to_res_map_name field")
        
        self.config = config

    def _construct_matrices_output_paths(self, csv_file_path: str, output_directory: str) -> Tuple[str, str]:
        """
        Construct and create output directory paths for the matrices based on the CSV file name.

        The method extracts the frame range from the CSV file name, creates the output
        directory accordingly, and constructs the file paths for both interactions and
        probabilities matrices.

        Args:
            csv_file_path (str): Path to the input CSV file.
            output_directory (str): Base directory where the output directories will be created.
        
        Returns:
            Tuple[str, str]: A tuple containing:
                - interactions_output_path (str): Path to save the interactions matrix.
                - probabilities_output_path (str): Path to save the probabilities matrix.
        """
        # extract the frames range from the csv file name
        csv_file_name = os.path.basename(csv_file_path)
        start_frame, end_frame = _util.frames_from_name(csv_file_name)

        # construct the output directory path
        container_dir_name = self.config["matrices_directory_name"].format(start=start_frame, end=end_frame)
        matrices_directory_path = os.path.join(output_directory, container_dir_name)

        # create the output directory
        os.makedirs(matrices_directory_path, exist_ok=True)

        # construct the paths for the future interactions and probabilities matrices 
        interactions_output_path = os.path.join(matrices_directory_path, self.config["interactions_matrix_name"])
        probabilities_output_path = os.path.join(matrices_directory_path, self.config["probabilities_matrix_name"])

        return interactions_output_path, probabilities_output_path

    @staticmethod
    def aggregate_energies(csv_file_path: str) -> pd.DataFrame:
        """
        Aggregate energy values from a CSV file by summing them for each residue index pair.

        The method reads the CSV file and groups the data by the "residue_i_index" and
        "residue_j_index" columns, summing the associated energy values.

        Args:
            csv_file_path (str): Path to the CSV file.
        
        Returns:
            pd.DataFrame: DataFrame with aggregated energy values.
        """
        # read the CSV file into a df
        df = pd.read_csv(csv_file_path)
        
        # group by the residue index columns, summing the energy values
        aggregated_energies = df.groupby(["residue_i_index", "residue_j_index"], as_index=False)["energy"].sum()
        return aggregated_energies

    @staticmethod
    def to_interactions_probablities_matrices(aggregated_energies: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert aggregated energies DataFrame to symmetric interactions and probabilities matrices.

        The method creates two matrices:
          - The interactions matrix contains absolute energy values with zero self-interactions.
          - The probabilities matrix is normalized per row, with the diagonal set to the smallest possible
            float value for safe logarithm operations.

        Args:
            aggregated_energies (pd.DataFrame): DataFrame containing aggregated energy values,
                with columns "residue_i_index", "residue_j_index", and "energy".
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - interactions_matrix (np.ndarray): The symmetric interactions matrix.
                - probabilities_matrix (np.ndarray): The normalized probabilities matrix.
        """
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
        """
        Aggregate energy values from a CSV file, convert them to matrices, and save the matrices.

        This method performs the following steps:
          1. Aggregates energies from the CSV file.
          2. Converts the aggregated energies into interactions and probabilities matrices.
          3. Constructs output file paths based on the CSV file name.
          4. Saves the matrices using NumPy's np.save function.
        
        Args:
            csv_file_path (str): Path to the CSV file.
            output_directory (str): Base directory for saving the output matrices.
        """
        aggregated_energies = ToMatricesConverter.aggregate_energies(csv_file_path)
        interactions_matrix, probaility_matrix = ToMatricesConverter.to_interactions_probablities_matrices(aggregated_energies)

        interactions_output_path, probabilities_output_path = self._construct_matrices_output_paths(csv_file_path, output_directory)
        np.save(interactions_output_path, interactions_matrix)
        np.save(probabilities_output_path, probaility_matrix)

    def process_multiple_csv_files(self, csv_files_to_process: List[str], output_directory: str) -> None:
        """
        Process multiple CSV files by aggregating, converting, and saving matrices for each file.

        Args:
            csv_files_to_process (List[str]): List of CSV file paths to process.
            output_directory (str): Base directory where the output matrices will be saved.
        """
        for file in csv_files_to_process:
            self.aggregate_convert_save(file, output_directory)

    def process_target_directory(self, target_directory_path: str,
                                 in_parallel: bool, allowed_memory_percentage_hint: Optional[float] = None,
                                 num_workers: Optional[int] = None, output_directory_path: Optional[str] = None,
                                 create_id_to_res_map: Optional[bool] = True) -> str:
        """
        Process all CSV files in a target directory by converting them to matrices.

        Depending on the `in_parallel` flag, the processing can be performed either in parallel or sequentially.

        Args:
            target_directory_path (str): Path to the directory containing CSV files.
            in_parallel (bool): Flag indicating whether to process files in parallel.
            allowed_memory_percentage_hint (Optional[float]): Hint for allowed memory usage percentage 
                when processing in parallel. Required if `in_parallel` is True.
            num_workers (Optional[int]): Number of worker processes for parallel processing.
                Required if `in_parallel` is True.
            output_directory_path (Optional[str]): Directory where the output matrices will be saved.
                If not provided, a default output directory is created.
        
        Returns:
            str: The path to the output directory where matrices are saved.
        
        Raises:
            ValueError: If parallel processing is requested but required parameters are missing.
        """
        output_directory = output_directory_path if output_directory_path else _util.create_output_dir(os.getcwd(), self.config["output_directory_name"])
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
        """
        Create a mapping from residue IDs to residue labels from a CSV file.

        This method reads the CSV file, sorts the rows by "residue_i_index", and removes duplicates.
        The resulting mapping (as a tuple) is then saved to a file in the specified output directory.

        Args:
            csv_file_path (str): Path to the CSV file.
            output_directory (str): Directory where the mapping file will be saved.
        
        Returns:
            Tuple[Any, ...]: A tuple containing the unique residue labels corresponding to each residue ID.
        """
        output_directory = output_directory_path if output_directory_path else _util.create_output_dir(os.getcwd(), self.config["output_directory_name"])
        df = pd.read_csv(csv_file_path)
        result = tuple(df.sort_values(by=["residue_i_index"]).drop_duplicates(subset=["residue_i_index"])["residue_i"])
        output_file_path = os.path.join(output_directory, self.config["id_to_res_map_name"])
        with open(output_file_path, "w") as output_file:
            output_file.write(str(result))

        return result
    

if __name__ == "__main__":
    pass
