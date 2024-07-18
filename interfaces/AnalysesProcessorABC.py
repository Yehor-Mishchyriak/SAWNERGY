from abc import ABC, abstractmethod, abstractstaticmethod

class AnalysesProcessorABC(ABC):
    """
    An abstract base class for processing analysis files, converting them to CSV format, and saving them to an output directory.

    Attributes:
        target_directory (str): The directory containing the analysis files.
        output_directory (str): The directory to save the output CSV files.
    """

    @abstractmethod
    def __init__(self, target_directory: str = None, output_directory: str = None) -> None:
        """
        Initialize the AnalysesProcessor with the target directory.

        Args:
            target_directory (str): The directory containing the analysis files.
            output_directory (str, optional): The directory to save the output CSV files.
        """

    @abstractstaticmethod
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
    def process_target_directory(self) -> str:
        """
        Process all .dat files in the target directory and convert them to CSV format.

        Returns:
            str: The path to the output directory containing the CSV files.

        Raises:
            Exception: If there is an error processing the files.
        """

    @abstractmethod
    def convert_to_csv(self, analysis_file: str, output_directory: str) -> None:
        """
        Convert a single analysis file to CSV format.

        This method must be compatible with the schema of the analysis_file.

        Args:
            analysis_file (str): The path to the analysis file.
            output_directory (str): The directory to save the output CSV file.

        Raises:
            Exception: If there is an error converting the file to CSV.
        """
