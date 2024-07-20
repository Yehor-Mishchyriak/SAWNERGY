#!/usr/bin/env python3

import os
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AnalysesProcessor:
    """
    A class to process analysis files, convert them to CSV format, and save them to an output directory.

    The output data is of the following format:
    - "Raw_Electrostatics_<time_when_created>" dir that contains "i-j.csv" files, where i is the start frame and j is the end frame indices.

    Attributes:
        target_directory (str): The directory containing the analysis files.
        output_directory (str): The directory to save the output CSV files.
    """

    def __init__(self, target_directory: str = None, output_directory: str = None) -> None:
        """
        Initialize the AnalysesProcessor with the target directory.

        Args:
            target_directory (str, optional): The directory containing the analysis files.
            output_directory (str, optional): The directory to save the output CSV files.
        """
        self.target_directory = target_directory if target_directory else os.getcwd()
        self.output_directory = self._create_output_dir(output_directory)

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
            output_directory = os.path.join(root_directory, f"Raw_Electrostatics_{current_time}")
            os.makedirs(output_directory, exist_ok=True)
            logging.info(f"Created output directory: {output_directory}")
            return output_directory
        except OSError as e:
            logging.error(f"Error creating output directory: {e}")
            raise

    def convert_to_csv(self, analysis_file: str) -> None:
        """
        Convert a single analysis file to CSV format.

        Args:
            analysis_file (str): The path to the analysis file.

        Raises:
            Exception: If there is an error converting the file to CSV.
        """
        try:
            lines = self._read_lines(analysis_file)
            output_file_name = os.path.basename(analysis_file).replace(".dat", ".csv")
            self._write_csv(lines, os.path.join(self.output_directory, output_file_name))
            logging.info(f"Converted {analysis_file} to CSV format")
        except Exception as e:
            logging.error(f"Error converting file {analysis_file} to CSV: {e}")
            raise

    def _sequential_processor(self) -> str:
        """
        Process all .dat files in the target directory sequentially and convert them to CSV format.

        Returns:
            str: The path to the output directory containing the CSV files.
        """
        logging.info("Starting sequential processing")
        try:
            for file in os.listdir(self.target_directory):
                if file.endswith(".dat"):
                    self.convert_to_csv(os.path.join(self.target_directory, file))
                    logging.info(f"Processed file {file} sequentially")
        except Exception as e:
            logging.error(f"Error processing files sequentially: {e}")
            raise

        logging.info("Sequential processing complete")
        return self.output_directory

    def _parallel_processor(self) -> str:
        """
        Process all .dat files in the target directory in parallel and convert them to CSV format.

        Returns:
            str: The path to the output directory containing the CSV files.
        """
        logging.info("Starting parallel processing")
        tasks = []
        with ProcessPoolExecutor() as executor:
            for file in os.listdir(self.target_directory):
                if file.endswith(".dat"):
                    tasks.append(executor.submit(self.convert_to_csv, os.path.join(self.target_directory, file)))
                    logging.info(f"Submitted task for file {file}")

            for future in as_completed(tasks):
                try:
                    future.result()  # This will re-raise any exception caught in the thread
                    logging.info("Processed file successfully in parallel")
                except Exception as e:
                    logging.error(f"Error processing file: {e}")
                    raise

        logging.info("Parallel processing complete")
        return self.output_directory

    def process_target_directory(self) -> str:
        """
        Process all .dat files in the target directory and convert them to CSV format.

        Returns:
            str: The path to the output directory containing the CSV files.

        Raises:
            Exception: If there is an error processing the files.
        """
        try:
            if __name__ == "__main__":
                logging.info("Using parallel processing")
                return self._parallel_processor()
            else:
                logging.info("Using sequential processing")
                return self._sequential_processor()
        except Exception as e:
            logging.error(f"Error processing target directory: {e}")
            raise

    @staticmethod
    def _read_lines(file_path: str) -> list:
        """
        Read lines from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            list: List of lines in the file.

        Raises:
            IOError: If there is an error reading the file.
        """
        try:
            with open(file_path, "r") as file:
                lines = file.readlines()[1::]
                logging.info(f"Read lines from file: {file_path}")
                return lines
        except IOError as e:
            logging.error(f"Error reading file {file_path}: {e}")
            raise

    @staticmethod
    def _write_csv(lines: list, output_file_path: str) -> None:
        """
        Write the CSV file with the formatted data.

        Args:
            lines (list): List of lines to be written.
            output_file_path (str): The path to the output CSV file.

        Raises:
            IOError: If there is an error writing the file.
        """
        header = "residue_i,residue_i_index,residue_j,residue_j_index,energy\n"
        try:
            with open(output_file_path, "w") as output_file:
                output_file.write(header)
                for line in lines:
                    parsed_line = AnalysesProcessor._parse_line(line)
                    output_file.write(parsed_line)
                logging.info(f"Written CSV file to: {output_file_path}")
        except IOError as e:
            logging.error(f"Error writing file {output_file_path}: {e}")
            raise

    @staticmethod
    def _parse_line(line: str) -> str:
        """
        Parse a line of the analysis file and return it in CSV format.

        Args:
            line (str): A line from the analysis file.

        Returns:
            str: The parsed line in CSV format.
        """
        res_atm_A, _, _, res_atm_B, _, _, _, energy = line.split()
        res_A, res_A_index = AnalysesProcessor._split_residue(res_atm_A)
        res_B, res_B_index = AnalysesProcessor._split_residue(res_atm_B)
        return f"{res_A},{res_A_index},{res_B},{res_B_index},{energy}\n"

    @staticmethod
    def _split_residue(res_atm: str) -> tuple:
        """
        Split a residue string into its components.

        Args:
            res_atm (str): Residue string in the format "residue_index@atom".

        Returns:
            tuple: Residue and residue index.
        """
        residue = res_atm.split("_")[0]
        residue_index = res_atm.split("_")[1].split("@")[0]
        return residue, residue_index


def main():
    """
    Main function to execute the processor.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Target directory with optional output directory")
    parser.add_argument('target_directory', type=str, help='The directory containing the analysis files')
    parser.add_argument('--output_directory', type=str, default=None, help='The directory to save the output CSV files')

    args = parser.parse_args()

    try:
        analyses_processor = AnalysesProcessor(args.target_directory, args.output_directory)
        print(analyses_processor.process_target_directory())
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
