import os
from datetime import datetime
from subprocess import run, CalledProcessError
from interfaces.FramesAnalyserABC import FramesAnalyserABC
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FramesElectrostaticsAnalyser(FramesAnalyserABC):
    """
    A class to process frames from topology and trajectory files using cpptraj.

    Attributes:
        output_directory (str): The directory where output files will be saved.
        number_frames (int): Total number of frames to be processed.
        start_frame (int): The starting frame for processing.
        batch_size (int): The number of frames to process in each batch.
        number_batches (int): The number of full batches to process.
        residual_frames (int): The number of frames remaining after full batches are processed.
        _analyse (function): The function to run the cpptraj analysis.
    """

    def __init__(self, topology_file: str, trajectory_file: str, number_frames: int,
                 cpptraj_analysis_command: str, cpptraj_output_type: str, start_frame: int = 1,
                 batch_size: int = 1, in_one_batch: bool = False, output_directory: str = None) -> None:
        """
        Initialize the FramesElectrostaticsAnalyser with the given parameters and set up the output directory.

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
        try:
            self.output_directory = FramesElectrostaticsAnalyser._create_output_dir(output_directory)
            self._analyse = self._analysis_function(topology_file, trajectory_file, cpptraj_analysis_command, cpptraj_output_type, self.output_directory)
            self.number_frames = number_frames
            self.start_frame = start_frame
            self.batch_size = number_frames if in_one_batch else batch_size
            self.number_batches, self.residual_frames = divmod(self.number_frames, self.batch_size)
        except Exception as e:
            logging.error(f"Error initializing FramesElectrostaticsAnalyser: {e}")
            raise

    @staticmethod
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
        try:
            current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
            root_directory = output_directory if output_directory else os.getcwd()
            output_directory = os.path.join(root_directory, f"OUTPUT_{current_time}")
            os.makedirs(output_directory, exist_ok=True)
            return output_directory
        except OSError as e:
            logging.error(f"Error creating output directory: {e}")
            raise

    @staticmethod
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
        def inner(start_frame: int, end_frame: int):
            nonlocal topology_file, trajectory_file, cpptraj_analysis_command, cpptraj_output_type, output_directory
            try:
                command = (
                    f"echo \"parm {topology_file} trajin {trajectory_file} "
                    f"{start_frame} {end_frame} {cpptraj_analysis_command} "
                    f"{cpptraj_output_type} {output_directory}/{start_frame}_{end_frame}.dat run\" | cpptraj"
                )
                run(command, check=True, shell=True)
            except CalledProcessError as e:
                logging.error(f"Error running cpptraj for frames {start_frame} to {end_frame}: {e}")
                raise

        return inner

    def __sequential_processor(self):
        """
        Process the frames sequentially in batches.
        """
        current_start_frame = self.start_frame
        current_end_frame = self.start_frame + self.batch_size - 1

        for _ in range(self.number_batches):
            self._analyse(current_start_frame, current_end_frame)
            current_start_frame += self.batch_size
            current_end_frame += self.batch_size

        if self.residual_frames > 0:
            self._analyse(current_start_frame, current_start_frame + self.residual_frames - 1)

    def __parallel_processor(self) -> str:
        """
        Process the frames in parallel using a process pool.

        Returns:
            str: The directory where the output files are saved.
        """
        current_start_frame = self.start_frame
        current_end_frame = self.start_frame + self.batch_size - 1

        tasks = list()
        with ProcessPoolExecutor() as executor:
            for _ in range(self.number_batches):
                tasks.append(executor.submit(self._analyse, current_start_frame, current_end_frame))
                current_start_frame += self.batch_size
                current_end_frame += self.batch_size

            if self.residual_frames > 0:
                tasks.append(executor.submit(self._analyse, current_start_frame, current_start_frame + self.residual_frames - 1))

            for future in as_completed(tasks):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error processing batch: {e}")

        return self.output_directory

    def analyse_frames(self) -> str:
        """
        Analyse the frames by processing them in batches.

        Returns:
            str: The directory where the output files are saved.
        """
        if __name__ == "__main__":
            result = self.__parallel_processor()
        else:
            result = self.__sequential_processor()

        return result

    @staticmethod
    def exctract_residues_from_pdb(pdb_file: str, save_output=False, output_directory = None) -> dict:
        # This is a helper method that adds some functionality to the class
        # Needs adding a docstring and error handling
        # Needs checking whether the pdb_file is actually a pdb (does it have .pdb extension)
        residues = dict()
        with open(pdb_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                try:
                    _, _, _, residue, index, _, _, _, _, _ = line.split()
                except ValueError: # in case the line being parsed is of a different format
                    continue
                residue = residue.strip()
                index = int(index.strip())
                residues[index] = residue

        if save_output:
            if output_directory is None:
                output_directory = os.getcwd()
            path = os.path.join(output_directory, f"{pdb_file}_residues.py")
            with open(path, 'w') as output_file:
                print(f"residues = {residues}", file=output_file)

        return residues


def main():
    """
    Main function to parse arguments and run the frame processing.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Process topology and trajectory files with optional parameters")
    parser.add_argument('topology_file', type=str, help='Path to the topology file')
    parser.add_argument('trajectory_file', type=str, help='Path to the trajectory file')
    parser.add_argument('number_frames', type=int, help='Total number of frames')
    parser.add_argument('cpptraj_analysis_command', type=str, help='cpptraj analysis command')
    parser.add_argument('cpptraj_output_type', type=str, help='cpptraj output type')
    parser.add_argument('--start_frame', type=int, default=1, help='The starting frame (default: 1)')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size (default: 1)')
    parser.add_argument('--in_one_batch', action='store_true', help='Process all frames in one batch')
    parser.add_argument('--output_directory', type=str, default=None, help='Directory to save the output files')

    args = parser.parse_args()

    try:
        frame_analyser = FramesElectrostaticsAnalyser(args.topology_file, args.trajectory_file, args.number_frames, 
                                        args.cpptraj_analysis_command, args.cpptraj_output_type, args.start_frame, 
                                        args.batch_size, args.in_one_batch, args.output_directory)
        frame_analyser.analyse_frames()
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
