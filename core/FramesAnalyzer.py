#!AllostericPathwayAnalyzer/venv/bin/python3

# external imports
import os
import inspect
from subprocess import run, CalledProcessError
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Tuple

# local imports
import core
from .util import construct_batch_sequence, process_elementwise, init_error_handler_n_logger, generic_error_handler_n_logger


class FramesAnalyzer:
    """
    A class for exctracting interatomic interaction energies from an MD trajectory frames using cpptraj.

    Attributes:
        topology_file (str): Path to the MD topology file.
        trajectory_file (str): Path to the MD trajectory file.
        cpptraj_analysis_command (str): cpptraj command specifying the analysis type.
        cpptraj_output_type (str): cpptraj output type (for example 'avgout').
        batches (list): Sequence of frame batches to be processed.
        output_directory (str): Path to the directory where cpptraj output is saved.
    """

    @init_error_handler_n_logger(core.network_construction_logger)
    def __init__(self, topology_file: str, trajectory_file: str,
                number_frames: int,
                cpptraj_analysis_command: str, cpptraj_output_type: str,
                batch_size: int = 1, in_one_batch: bool = False, output_directory_path: Union[str, None] = None) -> None:
        """
        Initializes the FramesAnalyzer class.

        Args:
            topology_file (str): Path to the MD topology file.
            trajectory_file (str): Path to the MD trajectory file.
            number_frames (int): Number of frames in the MD trajectory.
            cpptraj_analysis_command (str): cpptraj command specifying the analysis type.
            cpptraj_output_type (str): cpptraj output type (for example 'avgout').
            batch_size (int, optional): Number of frames per batch. Defaults to 1.
            in_one_batch (bool, optional): Whether to process all frames in one batch. Defaults to False.
            output_directory_path (str, optional): Path to the directory for saving cpptraj output. If None, a directory will be created automatically.
        """
        # MD simulation data import
        if not os.path.exists(topology_file):
            core.network_construction_logger.critical(f"Topology file does not exist: {topology_file}")
            raise FileNotFoundError(f"Topology file does not exist: {topology_file}")
        self.topology_file = topology_file

        if not os.path.exists(trajectory_file):
            core.network_construction_logger.critical(f"Trajectory file does not exist: {trajectory_file}")
            raise FileNotFoundError(f"Trajectory file does not exist: {trajectory_file}")
        self.trajectory_file = trajectory_file

        # cpptraj parameters
        self.cpptraj_analysis_command = cpptraj_analysis_command
        self.cpptraj_output_type = cpptraj_output_type

        # frame batches
        batch_size = number_frames if in_one_batch else batch_size
        self.batches = construct_batch_sequence(number_frames, batch_size)

        # cpptraj output directory
        self.output_directory = output_directory_path if output_directory_path else core.create_output_dir(
                                                        core.root_config["GLOBAL"]["output_directory_path"],
                                                        core.root_config["FramesAnalyzer"]["output_directory_name"])

    @generic_error_handler_n_logger(core.network_construction_logger, exclude_logging_exceptions=(CalledProcessError,))
    def _run_cpptraj(self, start_end: Tuple[int,int]) -> None:
        """
        Runs the cpptraj command, specified at the class initialisation, on a batch of frames.

        Args:
            start_end (tuple): A tuple containing the start and end frame indices for the batch.
        """
        start_frame, end_frame = start_end
        output_file_name = core.root_config["FramesAnalyzer"]["start_end_frames_dependent_analysis_file_name"].format(start_frame, end_frame)
        output_file_path = os.path.join(self.output_directory, output_file_name)

        command = f"""echo \"parm {self.topology_file}
                    trajin {self.trajectory_file} {start_frame} {end_frame}
                    {self.cpptraj_analysis_command} {self.cpptraj_output_type} {output_file_path} run\" | cpptraj > /dev/null 2>&1"""
        
        try:
            core.network_construction_logger.info(f"Began processing {start_frame}-{end_frame} frame(s) in {inspect.currentframe().f_code.co_name} function.")
            run(command, check=True, shell=True)
            core.network_construction_logger.info(f"Successfully processed the frame(s).")
        except CalledProcessError as e:
            core.network_construction_logger.error(f"cpptraj invoked by {inspect.currentframe().f_code.co_name} failed for frame(s) {start_frame}-{end_frame}: {e}")
            raise
    
    @generic_error_handler_n_logger(core.network_construction_logger)
    def analyse_frames(self) -> str:
        """
        Runs the cpptraj command on all the batches of frames of an MD trajectory. 

        Is run in parallel if the script is executed as the main module, or sequentially otherwise.

        Returns:
            str: Path to the output directory containing the cpptraj results.
        """
        if __name__ == "__main__":
            core.network_construction_logger.info(f"Began processing frame batches in parallel.")
            process_elementwise(in_parallel=True, Executor=ThreadPoolExecutor)(self.batches, self._run_cpptraj)
        else:
            core.network_construction_logger.info(f"Began processing frame batches sequentially.")
            process_elementwise(in_parallel=False)(self.batches, self._run_cpptraj)

        core.network_construction_logger.info(f"Successfully processed all the batches.")
        return self.output_directory


def main():
    """
    Main function to parse arguments and run the frame processing.
    """
    import argparse
    import shutil

    # Check if cpptraj is available
    if shutil.which('cpptraj') is None:
        core.network_construction_logger.critical("cpptraj is not installed or not found in the system PATH.")
        print("Error: cpptraj is required but not installed or not found in PATH.")
        exit(1)

    parser = argparse.ArgumentParser(description="Process topology and trajectory files with optional parameters")
    parser.add_argument('topology_file', type=str, help='Path to the topology file')
    parser.add_argument('trajectory_file', type=str, help='Path to the trajectory file')
    parser.add_argument('number_frames', type=int, help='Total number of frames')
    parser.add_argument('cpptraj_analysis_command', type=str, help='cpptraj analysis command')
    parser.add_argument('cpptraj_output_type', type=str, help='cpptraj output type')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size (default: 1)')
    parser.add_argument('--in_one_batch', action='store_true', help='Process all frames in one batch')
    parser.add_argument('--output_directory', type=str, default=None, help='Directory to save the output files')

    args = parser.parse_args()

    frame_analyzer = FramesAnalyzer(args.topology_file, args.trajectory_file, args.number_frames,
                                        args.cpptraj_analysis_command, args.cpptraj_output_type,
                                        args.batch_size, args.in_one_batch, args.output_directory)
    
    # return the output directory to the stdout
    print(frame_analyzer.analyse_frames())


if __name__ == "__main__":
    main()
