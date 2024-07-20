#!/usr/bin/env python3.9

import os
import logging
import shutil
from typing import Dict
from datetime import datetime
from subprocess import run, CalledProcessError
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FramesAnalyzer:
    """
    A class to process frames from topology and trajectory files using cpptraj.

    * The output data is of the following format:
    * "Processed_Frames_<time_when_created>" dir that contains "i-j.dat" files, where i is the start frame and j is the end frame indices

    Attributes:
        output_directory (str): The directory where output files will be saved.
        number_frames (int): Total number of frames to be processed.
        start_frame (int): The starting frame for processing.
        batch_size (int): The number of frames to process in each batch.
        number_batches (int): The number of full batches to process.
        residual_frames (int): The number of frames remaining after full batches are processed.
        cpptraj_analysis_command (str): The cpptraj command to be executed for analysis.
        cpptraj_output_type (str): The type of output expected from cpptraj.
    """

    def __init__(self, topology_file: str, trajectory_file: str, number_frames: int,
                 cpptraj_analysis_command: str, cpptraj_output_type: str, start_frame: int = 1,
                 batch_size: int = 1, in_one_batch: bool = False, output_directory: str = None) -> None:
        """
        Initialize the FramesAnalyzer with the given parameters and set up the output directory.

        Args:
            topology_file (str): Path to the topology file.
            trajectory_file (str): Path to the trajectory file.
            number_frames (int): Total number of frames to process.
            cpptraj_analysis_command (str): The cpptraj command to be executed for analysis.
            cpptraj_output_type (str): The type of output expected from cpptraj.
            start_frame (int, optional): The starting frame for processing. Defaults to 1.
            batch_size (int, optional): The number of frames to process in each batch. Defaults to 1.
            in_one_batch (bool, optional): Process all frames in one batch. Defaults to False.
            output_directory (str, optional): The directory where output files will be saved. Defaults to the current working directory.
        """
        try:
            self.topology_file = topology_file
            self.trajectory_file = trajectory_file
            self.number_frames = number_frames
            self.cpptraj_analysis_command = cpptraj_analysis_command
            self.cpptraj_output_type = cpptraj_output_type
            self.start_frame = start_frame
            self.batch_size = number_frames if in_one_batch else batch_size
            self.number_batches, self.residual_frames = divmod(self.number_frames, self.batch_size)
            self.output_directory = self._create_output_dir(output_directory)
        except Exception as e:
            logging.error(f"Error initializing FramesAnalyzer: {e}")
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
            output_directory = os.path.join(root_directory, f"Processed_Frames_{current_time}")
            os.makedirs(output_directory, exist_ok=True)
            logging.info(f"Created output directory: {output_directory}")
            return output_directory
        except OSError as e:
            logging.error(f"Error creating output directory: {e}")
            raise

    def _run_cpptraj(self, start_frame: int, end_frame: int) -> None:
        """
        Run cpptraj with the given parameters.

        Args:
            start_frame (int): The starting frame for cpptraj analysis.
            end_frame (int): The ending frame for cpptraj analysis.

        Raises:
            CalledProcessError: If cpptraj command fails.
        """
        try:
            output_file_path = os.path.join(self.output_directory, f"{start_frame}-{end_frame}.dat")
            command = f"""echo \"parm {self.topology_file}
                        trajin {self.trajectory_file} {start_frame} {end_frame}
                        {self.cpptraj_analysis_command} {self.cpptraj_output_type} {output_file_path} run\" | cpptraj > /dev/null 2>&1"""
            run(command, check=True, shell=True)
            logging.info(f"Successfully processed frames {start_frame} to {end_frame}")
        except CalledProcessError as e:
            logging.error(f"Error running cpptraj for frames {start_frame} to {end_frame}: {e}")
            raise

    def _sequential_processor(self) -> str:
        """
        Process the frames sequentially in batches.

        Returns:
            str: The directory where the output files are saved.
        """
        logging.info("Starting sequential processing")
        current_start_frame = self.start_frame
        current_end_frame = self.start_frame + self.batch_size - 1

        for _ in range(self.number_batches):
            try:
                self._run_cpptraj(current_start_frame, current_end_frame)
                logging.info(f"Processed frames {current_start_frame} to {current_end_frame} sequentially")
            except Exception as e:
                logging.error(f"Error processing frames {current_start_frame} to {current_end_frame} sequentially: {e}")
            current_start_frame += self.batch_size
            current_end_frame += self.batch_size

        if self.residual_frames > 0:
            try:
                self._run_cpptraj(current_start_frame, current_start_frame + self.residual_frames - 1)
                logging.info(f"Processed residual frames {current_start_frame} to {current_start_frame + self.residual_frames - 1} sequentially")
            except Exception as e:
                logging.error(f"Error processing residual frames {current_start_frame} to {current_start_frame + self.residual_frames - 1} sequentially: {e}")

        logging.info("Sequential processing complete")
        return self.output_directory

    def _parallel_processor(self) -> str:
        """
        Process the frames in parallel using a process pool.

        Returns:
            str: The directory where the output files are saved.
        """
        logging.info("Starting parallel processing")
        current_start_frame = self.start_frame
        current_end_frame = self.start_frame + self.batch_size - 1

        tasks = []
        with ProcessPoolExecutor() as executor:
            for _ in range(self.number_batches):
                tasks.append(executor.submit(self._run_cpptraj, current_start_frame, current_end_frame))
                current_start_frame += self.batch_size
                current_end_frame += self.batch_size

            if self.residual_frames > 0:
                tasks.append(executor.submit(self._run_cpptraj, current_start_frame, current_start_frame + self.residual_frames - 1))

            for future in as_completed(tasks):
                try:
                    future.result()
                    logging.info("Batch processed successfully in parallel")
                except Exception as e:
                    logging.error(f"Error processing batch in parallel: {e}")

        logging.info("Parallel processing complete")
        return self.output_directory

    def analyse_frames(self) -> str:
        """
        Analyse the frames by processing them in batches.

        Returns:
            str: The directory where the output files are saved.
        """
        if __name__ == "__main__":
            logging.info("Using parallel processing")
            return self._parallel_processor()
        else:
            logging.info("Using sequential processing")
            return self._sequential_processor()

    @staticmethod
    def extract_residues_from_pdb(pdb_file: str, save_output: bool = False, output_directory: str = None) -> Dict[int, str]:
        """
        Extract residues from a PDB file.

        Args:
            pdb_file (str): Path to the PDB file.
            save_output (bool, optional): Whether to save the output to a file. Defaults to False.
            output_directory (str, optional): Directory to save the output file. Defaults to None.

        Returns:
            Dict[int, str]: A dictionary of residue indices and residue names.

        Raises:
            ValueError: If the file does not have a .pdb extension.
        """
        if not pdb_file.endswith('.pdb'):
            raise ValueError("The file provided is not a PDB file.")

        residues = {}
        try:
            with open(pdb_file, "r") as file:
                lines = file.readlines()
                for line in lines:
                    try:
                        _, _, _, residue, index, _, _, _, _, _ = line.split()
                        residue = residue.strip()
                        index = int(index.strip())
                        residues[index] = residue
                    except ValueError:  # in case the line being parsed is of a different format
                        continue
        except FileNotFoundError as e:
            logging.error(f"PDB file not found: {e}")
            raise
        except Exception as e:
            logging.error(f"Error extracting residues: {e}")
            raise

        if save_output:
            if output_directory is None:
                output_directory = os.getcwd()
            output_file_path = os.path.join(output_directory, f"{os.path.splitext(os.path.basename(pdb_file))[0]}_residues.py")
            try:
                with open(output_file_path, 'w') as output_file:
                    output_file.write(f"residues = {residues}")
                logging.info(f"Residue data saved to {output_file_path}")
            except IOError as e:
                logging.error(f"Error saving residue data: {e}")
                raise

        return residues


def main():
    """
    Main function to parse arguments and run the frame processing.
    """
    import argparse

    # Check if cpptraj is available
    if shutil.which('cpptraj') is None:
        logging.error("cpptraj is not installed or not found in the system PATH.")
        return

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
        frame_analyzer = FramesAnalyzer(args.topology_file, args.trajectory_file, args.number_frames,
                                        args.cpptraj_analysis_command, args.cpptraj_output_type, args.start_frame,
                                        args.batch_size, args.in_one_batch, args.output_directory)
        print(frame_analyzer.analyse_frames())  # output to the terminal
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
