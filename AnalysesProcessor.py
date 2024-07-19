import os
from datetime import datetime
from interfaces.AnalysesProcessorABC import AnalysesProcessorABC
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AnalysesProcessor(AnalysesProcessorABC):
    """
    A class to process analysis files, convert them to CSV format, and save them to an output directory.

    Attributes:
        target_directory (str): The directory containing the analysis files.
        output_directory (str): The directory to save the output CSV files.
    """

    def __init__(self, target_directory: str = None, output_directory=None) -> None:
        """
        Initialize the AnalysesProcessor with the target directory.

        Args:
            target_directory (str): The directory containing the analysis files.
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
    
    def convert_to_csv(self, analysis_file: str, output_directory: str) -> None:
        """
        Convert a single analysis file to CSV format.

        The analysis file must contain lines of data in the following format:
        
        SER_1@N     1 --         SER_1@HG    11 : EELEC=  7.32819e+00
        
        The line above is an example of the output of 
        "pairwise [<mask>] [<mask>] [cuteelec <ecut>]" command of cpptraj

        Args:
            analysis_file (str): The path to the analysis file.
            output_directory (str): The directory to save the output CSV file.

        Raises:
            Exception: If there is an error converting the file to CSV.
        """
        try:
            lines = self._read_lines(analysis_file)
            output_file_name = os.path.basename(analysis_file).replace(".dat", ".csv")
            self._write_csv(lines, os.path.join(output_directory, output_file_name))
            logging.info(f"Converted {analysis_file} to CSV format")
        except Exception as e:
            logging.error(f"Error converting file {analysis_file} to CSV: {e}")
            raise

    def __sequential_processor(self) -> str:
        """
        Process all .dat files in the target directory sequentially and convert them to CSV format.

        Returns:
            str: The path to the output directory containing the CSV files.
        """
        try:
            for file in os.listdir(self.target_directory):
                if file.endswith(".dat"):
                    path_to_file = os.path.join(self.target_directory, file)
                    self.convert_to_csv(path_to_file, self.output_directory)
            logging.info(f"Processed all .dat files sequentially in directory: {self.target_directory}")
        except Exception as e:
            logging.error(f"Error processing files sequentially: {e}")
            raise
        
        return self.output_directory

    def __parallel_processor(self) -> str:
        """
        Process all .dat files in the target directory in parallel and convert them to CSV format.

        Returns:
            str: The path to the output directory containing the CSV files.
        """
        tasks = []
        with ProcessPoolExecutor() as executor:
            for file in os.listdir(self.target_directory):
                if file.endswith(".dat"):
                    path_to_file = os.path.join(self.target_directory, file)
                    tasks.append(executor.submit(self.convert_to_csv, path_to_file, self.output_directory))
            logging.info(f"Submitted tasks for all .dat files in directory: {self.target_directory}")

            for future in as_completed(tasks):
                try:
                    future.result()  # This will re-raise any exception caught in the thread
                except Exception as e:
                    logging.error(f"Error processing file: {e}")
                    raise
        
        return self.output_directory

    def process_target_directory(self) -> str:
        # returns Raw_Electrostatics_{"%m-%d-%Y-%H-%M-%S"} directory where ({start_frame}-{end_frame}).csv files are stored
        """
        Process all .dat files in the target directory and convert them to CSV format.

        Returns:
            str: The path to the output directory containing the CSV files.

        Raises:
            Exception: If there is an error processing the files.
        """
        if __name__ == "__main__":
            result = self.__parallel_processor()
        else:
            result = self.__sequential_processor()
        
        return result
            
    @staticmethod
    def _read_lines(file_path: str) -> list:
        """
        Read every second line from a file starting from the third line.

        Args:
            file_path (str): The path to the file.

        Returns:
            list: List of every second line in the file starting from the third line.

        Raises:
            IOError: If there is an error reading the file.
        """
        try:
            with open(file_path, "r") as file:
                lines = list(map(str.strip, file.readlines()[1::]))
                logging.info(f"Read lines from file: {file_path}")
                return lines
        except IOError as e:
            logging.error(f"Error reading file {file_path}: {e}")
            raise

    @staticmethod
    def _write_csv(lines: list, output_file_path: str) -> None:
        """
        Write the CSV file with the formatted data.

        The output CSV file will have the following header:
        "residue_i,residue_i_index,residue_j,residue_j_index,energy"

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
        residue = res_atm[:res_atm.index("_")]
        residue_index = res_atm[res_atm.index("_") + 1:res_atm.index("@")]
        return residue, residue_index


def main():
    """
    Main function to execute the processor.
    """
    eap = AnalysesProcessor(target_directory="/Users/yehormishchyriak/Desktop/research_project/test_set", output_directory="/Users/yehormishchyriak/Desktop/research_project/output_files")
    eap.process_target_directory()


if __name__ == "__main__":
    main()
