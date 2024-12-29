#!AllostericPathwayAnalyzer/venv/bin/python3

# external imports
import os
import inspect
from concurrent.futures import ThreadPoolExecutor

# local imports
import core
from .util import read_lines, write_csv, process_elementwise, frames_from_name


class AnalysesProcessor:
    """
    A class for processing cpptraj .dat analysis files and converting them into .csv of the following schema:
    residue_i,residue_i_index,residue_j,residue_j_index,energy
    """
    def __init__(self, output_directory_path: str = None) -> None:
        """
        Initializes the AnalysesProcessor class.

        Args:
            output_directory_path (str, optional): Path to the directory for saving .csv output files. If None, a directory will be created automatically.

        Raises:
            KeyError: If the configuration file (root.json) is corrupt or incompatible with the class.
            FileNotFoundError: If a topology or a trajectory file does not exist.
            OSError: If there is an error creating or accessing the output directory.
        """
        try:
            self.output_directory = output_directory_path if output_directory_path else core.create_output_dir(
                                                        core.root_config["GLOBAL"]["output_directory_path"],
                                                        core.root_config["AnalysesProcessor"]["output_directory_name"])
            
            core.network_construction_logger.info(f"Successfully created the output directory for {self.__class__.__name__} class.")
            core.network_construction_logger.info(f"Successfully initialized {self.__class__.__name__} class.")

        except OSError as e:
            core.network_construction_logger.critical(
                f"Failed to create or access the output directory during {self.__class__.__name__} initialization. "
                f"Error: {e}"
            )
            raise OSError(
                f"An error occurred while creating or accessing the output directory for {self.__class__.__name__} class: {e}. "
                f"Check permissions and available disk space."
            ) from e

        except KeyError as e:
            core.network_construction_logger.critical(f"Corrupt/incompatible root configuration file: {e}")
            raise KeyError(f"Missing configuration key: {e}") from e

        except Exception as e:
            core.network_construction_logger.exception(f"Unexpected error occurred during {self.__class__.__name__} initialisation: {e}")
            raise

    def _cpptraj_to_csv(self, cpptraj_file_path: str) -> None:
        """
        Converts a cpptraj output file into a .csv file of the following schema:
        residue_i,residue_i_index,residue_j,residue_j_index,energy

        Args:
            cpptraj_file_path (str): Path to the cpptraj output file to process.

        Raises:
            FileNotFoundError: If the cpptraj file does not exist.
            ValueError: If the cpptraj file name format is invalid or the content is malformed.
            OSError: If there are issues reading the input file or writing the output CSV.
            Exception: For any unexpected errors during processing.
        """
        try:
            # read the cpptraj file
            contents = read_lines(cpptraj_file_path)

        except FileNotFoundError as e:
            core.network_construction_logger.error(f"{inspect.currentframe().f_code.co_name}: File not found: {cpptraj_file_path}")
            raise FileNotFoundError(f"Cpptraj file not found: {cpptraj_file_path}") from e
        
        except OSError as e:
            core.network_construction_logger.error(f"{inspect.currentframe().f_code.co_name}: Error reading file {cpptraj_file_path}: {e}")
            raise OSError(f"Error reading cpptraj file: {cpptraj_file_path}") from e
        
        except Exception as e:
            core.network_construction_logger.exception(f"{inspect.currentframe().f_code.co_name}: Unexpected error while reading file {cpptraj_file_path}: {e}")
            raise

        try:
            # extract frame range from the file name
            cpptraj_file_name = os.path.basename(cpptraj_file_path)
            start_frame, end_frame = frames_from_name(cpptraj_file_name)

        except AttributeError as e:
            core.network_construction_logger.error(f"{inspect.currentframe().f_code.co_name}: Invalid file name format: {cpptraj_file_name}")
            raise ValueError(f"Invalid cpptraj file name format: {cpptraj_file_name}. Expected format: 'startFrame-endFrame'.") from e
        
        except Exception as e:
            core.network_construction_logger.exception(f"{inspect.currentframe().f_code.co_name}: Unexpected error while parsing file name {cpptraj_file_name}: {e}")
            raise

        try:
            # construct the output CSV file path
            csv_file_name = core.root_config["AnalysesProcessor"]["start_end_frames_dependent_analysis_file_name"].format(start_frame, end_frame)
            csv_file_path = os.path.join(self.output_directory, csv_file_name)

        except KeyError as e:
            core.network_construction_logger.critical(f"{inspect.currentframe().f_code.co_name}: Missing configuration key: {e}")
            raise KeyError(f"Missing configuration key in root config: {e}") from e
        
        except Exception as e:
            core.network_construction_logger.exception(f"{inspect.currentframe().f_code.co_name}: Unexpected error constructing CSV file path: {e}")
            raise

        try:
            # write the contents to the CSV file
            write_csv(contents, csv_file_path)

        except OSError as e:
            core.network_construction_logger.error(f"{inspect.currentframe().f_code.co_name}: Error writing CSV file {csv_file_path}: {e}")
            raise OSError(f"Error writing to CSV file: {csv_file_path}") from e
        
        except Exception as e:
            core.network_construction_logger.exception(f"{inspect.currentframe().f_code.co_name}: Unexpected error while writing CSV file {csv_file_path}: {e}")
            raise

        core.network_construction_logger.info(f"{inspect.currentframe().f_code.co_name}: Successfully converted {cpptraj_file_path} to {csv_file_path}.")

    def process_target_directory(self, target_directory_path: str) -> str:
        """
        Processes all cpptraj output files in a target directory and converts them into .csv files
        with the following schema: residue_i,residue_i_index,residue_j,residue_j_index,energy
        
        Args:
            target_directory_path (str): Path to the directory containing cpptraj output files.

        Returns:
            str: Path to the directory where the processed CSV files are saved.
        """
        # assumes the directory contains only cpptraj output files!
        analysis_files_paths = (os.path.join(target_directory_path, file) for file in os.listdir(target_directory_path))

        try:

            if __name__ == "__main__":
                core.network_construction_logger.info(f"Began processing cpptraj output files in parallel.")
                process_elementwise(in_parallel=True, Executor=ThreadPoolExecutor)(analysis_files_paths, self._cpptraj_to_csv)
            else:
                core.network_construction_logger.info(f"Began processing cpptraj output files sequentially.")
                process_elementwise(in_parallel=False)(analysis_files_paths, self._cpptraj_to_csv)

            core.network_construction_logger.info(f"Successfully processed all the cpptraj output files.")
            return self.output_directory
        
        except Exception as e:
            core.network_construction_logger.exception(f"Unexpected error occurred during {inspect.currentframe().f_code.co_name} execution: {e}")
            raise
    

def main():
    """
    Main function to execute the processor.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Target directory with optional output directory")
    parser.add_argument('target_directory', type=str, help='The directory containing the analysis files')
    parser.add_argument('--output_directory', type=str, default=None, help='The directory to save the output CSV files to')

    args = parser.parse_args()

    analyses_processor = AnalysesProcessor(args.output_directory)
    print(analyses_processor.process_target_directory(args.target_directory))


if __name__ == "__main__":
    main()
