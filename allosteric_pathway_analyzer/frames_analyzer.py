# external imports
import os
from subprocess import run, SubprocessError
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

# local imports
from . import pkg_globals
from . import _util


class FramesAnalyzer:
    """
    A class for analyzing MD trajectory frames using the cpptraj command line tool.

    This class encapsulates configuration management and the execution of cpptraj commands
    on specified frame batches, optionally in parallel.
    """

    def __init__(self, cpptraj_abs_path: Optional[str] = None) -> None:
        """
        Initialize a FramesAnalyzer instance.

        Args:
            cpptraj_abs_path (Optional[str], optional): Absolute path to the cpptraj executable.
                If None, the system PATH is searched. Defaults to None.

        Raises:
            FileNotFoundError: If the cpptraj executable is not found or is inaccessible.
        """
        self.config = None
        self._cpptraj = _util.cpptraj_is_available_at(cpptraj_abs_path)
        if self._cpptraj is None:
            raise FileNotFoundError(f"cpptraj was not found or is inaccessible at {cpptraj_abs_path}")
        self.set_config(pkg_globals.default_config[self.__class__.__name__])

    @property
    def which_cpptraj(self) -> Optional[str]:
        """
        Get the path to the cpptraj executable.

        Returns:
            Optional[str]: The absolute path to cpptraj if available; otherwise, None.
        """
        return self._cpptraj

    def __repr__(self) -> str:
        """
        Return a string representation of the FramesAnalyzer instance.

        Returns:
            str: A string representation including the configuration and cpptraj path.
        """
        return f"{self.__class__.__name__}(config={self.config}, _cpptraj={self._cpptraj})"

    def set_config(self, config: dict) -> None:
        """
        Set the configuration for the FramesAnalyzer.

        The configuration must include the keys:
          - "output_directory_name"
          - "cpptraj_file_name"

        Args:
            config (dict): A configuration dictionary.

        Raises:
            ValueError: If required configuration keys are missing.
        """
        if "output_directory_name" not in config:
            raise ValueError(f"Invalid {self.__class__.__name__} config: missing output_directory_name field")
        if "cpptraj_file_name" not in config:
            raise ValueError(f"Invalid {self.__class__.__name__} config: missing cpptraj_file_name field")
        self.config = config

    def run_cpptraj(self, start_end: Tuple[int, int], topology_file: str, trajectory_file: str,
                     cpptraj_analysis_command: str, cpptraj_output_type: str, output_directory: str) -> None:
        """
        Execute the cpptraj command for a given frame range.

        This method builds and runs a command that pipes a series of cpptraj instructions
        (including the topology, trajectory, analysis command, and output configuration)
        to the cpptraj executable.

        Args:
            start_end (Tuple[int, int]): A tuple (start_frame, end_frame) specifying the frame range.
            topology_file (str): Path to the topology file.
            trajectory_file (str): Path to the trajectory file.
            cpptraj_analysis_command (str): The analysis command to run in cpptraj.
            cpptraj_output_type (str): The output type parameter for cpptraj.
            output_directory (str): Directory where the cpptraj output file will be saved.

        Raises:
            RuntimeError: If execution of the cpptraj command fails.
            KeyError: If a wrong 'cpptraj_file_name' format is passed.
        """
        start_frame, end_frame = start_end
        try:
            output_file_name = self.config["cpptraj_file_name"].format(start=start_frame, end=end_frame)
        except KeyError:
            raise KeyError(f"Wrong 'cpptraj_file_name' format. Expected a string containing {{\"start\"}}-{{\"end\"}}, instead got: {self.config["cpptraj_file_name"]}")
        output_file_path = os.path.join(output_directory, output_file_name)

        command = (
            f'echo "parm {topology_file}\n'
            f'trajin {trajectory_file} {start_frame} {end_frame}\n'
            f'{cpptraj_analysis_command} {cpptraj_output_type} {output_file_path} run" | '
            f'{self._cpptraj} > /dev/null 2>&1'
        )
        try:
            run(command, check=True, shell=True)
        except SubprocessError as e:
            raise RuntimeError(f"An exception occurred while executing the '{command}' command: {e}")

    def analyse_frames(self, topology_file: str, trajectory_file: str,
                       cpptraj_analysis_command: str, cpptraj_output_type: str,
                       number_frames: int, in_parallel: bool, batch_size: Optional[int] = 1,
                       in_one_batch: Optional[bool] = False, output_directory_path: Optional[str] = None) -> str:
        """
        Analyze frames by dividing them into batches and processing each batch with cpptraj.

        The method creates an output directory (if not provided), divides the total frames
        into batches (or uses a single batch if in_one_batch is True), and executes the cpptraj
        command on each batch. Batches can be processed sequentially or in parallel.

        Args:
            topology_file (str): Path to the topology file.
            trajectory_file (str): Path to the trajectory file.
            cpptraj_analysis_command (str): The analysis command for cpptraj.
            cpptraj_output_type (str): The output type parameter for cpptraj.
            number_frames (int): Total number of frames to process.
            in_parallel (bool): If True, process batches in parallel using ThreadPoolExecutor.
            batch_size (Optional[int], optional): Number of frames per batch. Defaults to 1.
            in_one_batch (Optional[bool], optional): If True, process all frames as a single batch. Defaults to False.
            output_directory_path (Optional[str], optional): Path to the output directory. If None, a new directory is created.

        Returns:
            str: The path to the output directory where results are stored.
        """
        output_directory = (
            output_directory_path if output_directory_path
            else _util.create_output_dir(os.getcwd(), self.config["output_directory_name"])
        )

        # If in_one_batch is True, treat all frames as one batch.
        effective_batch_size = number_frames if in_one_batch else batch_size
        batches = _util.construct_batch_sequence(number_frames, effective_batch_size)
        
        _util.process_elementwise(
            in_parallel=in_parallel,
            Executor=ThreadPoolExecutor,
            capture_output=False
        )(
            batches,
            self.run_cpptraj,
            topology_file,
            trajectory_file,
            cpptraj_analysis_command,
            cpptraj_output_type,
            output_directory
        )
        
        return output_directory


if __name__ == "__main__":
    pass
