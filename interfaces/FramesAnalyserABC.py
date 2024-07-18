from abc import ABC, abstractmethod

class FramesAnalyserABC(ABC):
    """
    An abstract base class for processing frames from topology and trajectory files using cpptraj.

    Attributes:
        output_directory (str): The directory where output files will be saved.
        number_frames (int): Total number of frames to be processed.
        start_frame (int): The starting frame for processing.
        batch_size (int): The number of frames to process in each batch.
        number_batches (int): The number of full batches to process.
        residual_frames (int): The number of frames remaining after full batches are processed.
        _analyse (function): The function to run the cpptraj analysis.
    """

    @abstractmethod
    def __init__(self, topology_file: str, trajectory_file: str, number_frames: int, 
                 cpptraj_analysis_command: str, cpptraj_output_type: str, start_frame: int = 1,
                 batch_size: int = 1, in_one_batch: bool = False, output_directory: str = None) -> None:
        """
        Initialize the FramesAnalyser with the given parameters and set up the output directory.

        Args:
            topology_file (str): Path to the topology file.
            trajectory_file (str): Path to the trajectory file.
            number_frames (int): Total number of frames to process.
            cpptraj_analysis_command (str): The cpptraj command to be executed for analysis.
            cpptraj_output_type (str): The type of output expected from cpptraj.
            start_frame (int, optional): The starting frame for processing. Defaults to 1.
            batch_size (int, optional): The number of frames to process in each batch. Defaults to 1.
            in_one_batch (bool, optional): Process all frames in one batch. Defaults to False.
            output_directory (str, optional): The directory where output files will be saved.
        """

    @abstractmethod
    def _create_output_dir(output_directory: str = None) -> str:
        """
        Create a unique output directory based on the current time.

        Args:
            output_directory (str, optional): The base directory to create the output directory in.

        Returns:
            str: The path to the created output directory.

        Raises:
            OSError: If there is an error creating the directory.
        """

    @abstractmethod
    def _analysis_function(topology_file: str, trajectory_file: str, cpptraj_analysis_command: str, cpptraj_output_type: str, output_directory: str):
        """
        Create the analysis function to run cpptraj with the given parameters.

        Args:
            topology_file (str): Path to the topology file.
            trajectory_file (str): Path to the trajectory file.
            cpptraj_analysis_command (str): The cpptraj command to be executed for analysis.
            cpptraj_output_type (str): The type of output expected from cpptraj.
            output_directory (str): The directory to save output files.

        Returns:
            function: A function to run cpptraj analysis for given frame ranges.
        """

    @abstractmethod
    def analyse_frames(self) -> str:
        """
        Analyse the frames by processing them in batches.

        Returns:
            str: The directory where the output files are saved.
        """
