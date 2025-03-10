# external imports
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# local imports
from . import pkg_globals
from . import _util


class AnalysesProcessor:
    """
    A processor for converting cpptraj output files into CSV format.

    This class provides methods for processing cpptraj output files either
    incrementally (by chunk) or immediately (by reading the entire file), and
    for processing all files in a target directory.
    """

    def __init__(self) -> None:
        """
        Initialize an AnalysesProcessor instance using the default configuration.
        
        The configuration is retrieved from pkg_globals.default_config using the class name.
        """
        self.global_config = None
        self.cls_config = None
        self.set_config(pkg_globals.default_config)
    
    def __repr__(self) -> str:
        """
        Return a string representation of the AnalysesProcessor instance.
        
        Returns:
            str: String representation including the current configuration.
        """
        return f"{self.__class__.__name__}(config={self.cls_config})"

    def set_config(self, config: dict) -> None:
        """
        Set the global and AnalysesProcessor-specific configuration.

        The configuration must include:
            - AnalysesProcessor
                - 'output_directory_name'
                - 'csv_file_header'
                - 'csv_file_name'
        
        Args:
            config (dict): A configuration dictionary
        
        Raises:
            ValueError: If the required configuration keys are missing
        """
        if self.__class__.__name__ not in config:
            raise ValueError(f"Invalid config: missing {self.__class__.__name__} sub-config")
        
        sub_config = config[self.__class__.__name__]
        if "output_directory_name" not in sub_config:
            raise ValueError(f"Invalid {self.__class__.__name__} sub-config: missing output_directory_name field")
        if "csv_file_header" not in sub_config:
            raise ValueError(f"Invalid {self.__class__.__name__} sub-config: missing csv_file_header field")
        if "csv_file_name" not in sub_config:
            raise ValueError(f"Invalid {self.__class__.__name__} sub-config: missing csv_file_name field")
        
        self.global_config = config
        self.cls_config = config[self.__class__.__name__]

    def _construct_csv_file_path(self, cpptraj_file_path: str, output_directory: str) -> str:
        """
        Construct the output CSV file path based on the cpptraj file name and configuration.
        
        This method extracts the frame range from the cpptraj file name and uses the
        'csv_file_name' template from the configuration to build the CSV file name.
        
        Args:
            cpptraj_file_path (str): Full path to the cpptraj output file.
            output_directory (str): Directory where the CSV file should be created.
        
        Returns:
            str: The full path to the output CSV file.
        """
        # Extract frame range from the file name.
        cpptraj_file_name = os.path.basename(cpptraj_file_path)
        start_frame, end_frame = _util.frames_from_name(cpptraj_file_name)

        # Construct the CSV file name using the configuration template.
        try:
            csv_file_name = self.cls_config["csv_file_name"].format(start=start_frame, end=end_frame)
        except KeyError:
            raise KeyError(f"Wrong 'csv_file_name' format. Expected a string containing {{\"start\"}}-{{\"end\"}}, instead got: {self.cls_config['cpptraj_file_name']}")
        csv_file_path = os.path.join(output_directory, csv_file_name)

        return csv_file_path

    def cpptraj_to_csv_incrementally(self, cpptraj_file_path: str, output_directory: str,
                                     allowed_memory_percentage_hint: float, num_workers: int) -> None:
        """
        Convert a cpptraj output file to CSV format incrementally (by processing file chunks).
        
        This method writes the CSV header, then processes the cpptraj file in chunks
        (to keep memory usage low) and appends each chunk to the CSV file.
        
        Args:
            cpptraj_file_path (str): Path to the cpptraj output file.
            output_directory (str): Directory where the CSV file will be created.
            allowed_memory_percentage_hint (float): Fraction (between 0 and 1) of available memory to use throughout the data processing.
            num_workers (int): Number of workers to use for the data processing
        """
        csv_file_path = self._construct_csv_file_path(cpptraj_file_path, output_directory)
        # Write the CSV header to the output file.
        _util.write_csv_header(self.cls_config["csv_file_header"], csv_file_path)
        # Process the cpptraj file in chunks and append each chunk to the CSV file.
        for chunk in _util.chunked_file(cpptraj_file_path, allowed_memory_percentage_hint, num_workers):
            _util.append_csv_from_cpptraj_electrostatics(chunk, csv_file_path)

    def cpptraj_to_csv_immediately(self, cpptraj_file_path: str, output_directory: str) -> None:
        """
        Convert a cpptraj output file to CSV format immediately (by reading the whole file).
        
        This method reads the entire cpptraj file at once and writes its content to a CSV file.
        
        Args:
            cpptraj_file_path (str): Path to the cpptraj output file.
            output_directory (str): Directory where the CSV file will be created.
        """
        csv_file_path = self._construct_csv_file_path(cpptraj_file_path, output_directory)
        # Read the entire cpptraj file.
        contents = _util.read_lines(cpptraj_file_path)
        # Write the CSV file with the provided header and contents.
        _util.write_csv_from_cpptraj_electrostatics(self.cls_config["csv_file_header"], contents, csv_file_path)

    def process_target_directory(self, target_directory_path: str,
                                in_parallel: bool, allowed_memory_percentage_hint: Optional[float] = None,
                                num_workers: Optional[int] = None, output_directory_path: Optional[str] = None) -> str:
        """
        Process all cpptraj output files in a target directory and convert them to CSV format.

        Depending on the 'in_parallel' flag, this method will either process the files concurrently using a ThreadPoolExecutor
        or sequentially. For parallel processing with incremental (chunked) conversion, both 'allowed_memory_percentage_hint'
        and 'num_workers' must be provided.

        Args:
            target_directory_path (str): Directory containing cpptraj output files.
            in_parallel (bool): If True, process files concurrently.
            allowed_memory_percentage_hint (float): Memory usage hint (as a fraction between 0 and 1) for incremental processing.
                Required if in_parallel is True.
            num_workers (int): Number of worker threads for parallel processing.
                Required if in_parallel is True.
            output_directory_path (Optional[str], optional): Directory where CSV files will be saved.
                If None, a new directory is created using the configured output_directory_name.

        Returns:
            str: The path to the output directory where CSV files have been created.

        Raises:
            ValueError: If in_parallel is True and either 'num_workers' or 'allowed_memory_percentage_hint' is not provided.
        """
        # Determine the output directory.
        output_directory = output_directory_path if output_directory_path else _util.create_output_dir(os.getcwd(), self.cls_config["output_directory_name"])
        # List all files in the target directory (assumes they are all cpptraj output files).
        analysis_file_paths = [os.path.join(target_directory_path, file) for file in os.listdir(target_directory_path)]
        
        if in_parallel:
            if num_workers is None:
                raise ValueError("If in_parallel=True, num_workers parameter must be provided")
            if allowed_memory_percentage_hint is None:
                raise ValueError("If in_parallel=True, allowed_memory_percentage_hint parameter must be provided")
            
            _util.process_elementwise(
                in_parallel=True,
                Executor=ThreadPoolExecutor,
                capture_output=False,
                max_workers=num_workers
            )(
                analysis_file_paths,
                self.cpptraj_to_csv_incrementally,
                output_directory,
                allowed_memory_percentage_hint,
                num_workers
            )
        else:
            _util.process_elementwise(
                in_parallel=False,
                capture_output=False
            )(
                analysis_file_paths,
                self.cpptraj_to_csv_immediately,
                output_directory
            )
        return output_directory


if __name__ == "__main__":
    pass
