# external imports
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# local imports
from . import pkg_globals
from . import _util


class AnalysesProcessor:

    def __init__(self) -> None:
        self.global_config = None
        self.cls_config = None
        self.set_config(pkg_globals.default_config)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.cls_config})"

    def set_config(self, config: dict) -> None:
        self.global_config = config
        self.cls_config = config[self.__class__.__name__]

    def _construct_csv_file_path(self, cpptraj_file_path: str, output_directory: str) -> str:
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
        csv_file_path = self._construct_csv_file_path(cpptraj_file_path, output_directory)
        # Write the CSV header to the output file.
        _util.write_csv_header(self.cls_config["csv_file_header"], csv_file_path)
        # Process the cpptraj file in chunks and append each chunk to the CSV file.
        for chunk in _util.chunked_file(cpptraj_file_path, allowed_memory_percentage_hint, num_workers):
            _util.append_csv_from_cpptraj_electrostatics(chunk, csv_file_path)

    def cpptraj_to_csv_immediately(self, cpptraj_file_path: str, output_directory: str) -> None:
        csv_file_path = self._construct_csv_file_path(cpptraj_file_path, output_directory)
        # Read the entire cpptraj file.
        contents = _util.read_lines(cpptraj_file_path)
        # Write the CSV file with the provided header and contents.
        _util.write_csv_from_cpptraj_electrostatics(self.cls_config["csv_file_header"], contents, csv_file_path)

    def process_target_directory(self, target_directory_path: str,
                                in_parallel: bool, allowed_memory_percentage_hint: Optional[float] = None,
                                num_workers: Optional[int] = None, output_directory_path: Optional[str] = None) -> str:
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
