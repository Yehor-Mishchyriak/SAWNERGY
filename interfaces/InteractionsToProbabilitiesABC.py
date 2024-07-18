from abc import ABC, abstractmethod
from numpy import array

class InteractionsToProbabilitiesABC(ABC):
    """
    An abstract base class to process interaction matrices, convert them to probability matrices, and save them to an output directory.

    Attributes:
        target_directory (str): The directory containing the interaction matrix files.
        output_directory (str): The directory to save the output probability matrix files.
    """

    @abstractmethod
    def __init__(self, target_directory: str = None, output_directory: str = None) -> None:
        """
        Initialize the InteractionsToProbabilities with the target and output directories.

        Args:
            target_directory (str, optional): The directory containing the interaction matrix files.
            output_directory (str, optional): The directory to save the output probability matrix files.
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
    def convert_to_probabilities(interactions_matrix: array) -> array:
        """
        Convert an interaction matrix to a probability matrix using a utility function.

        Args:
            interactions_matrix (np.array): The interaction matrix to convert.

        Returns:
            np.array: The converted probability matrix.
        """

    @abstractmethod
    def process_target_directory(self) -> list:
        """
        Process the target directory of interactions matrices.

        Returns:
            list: List of converted probability matrices.
        """
