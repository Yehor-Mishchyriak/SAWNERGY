from .frames_analyzer import FramesAnalyzer
from .analyses_processor import AnalysesProcessor
from .to_matrices_converter import ToMatricesConverter
from .protein import Protein
from .network_analyzer import NetworkAnalyzer
import shutil

def construct_network(fa: FramesAnalyzer, ap: AnalysesProcessor, tmc: ToMatricesConverter,
                      fa_kwargs: dict, ap_kwargs: dict, tmc_kwargs: dict):
    """
    Construct the network by sequentially processing frames, analyses, and matrix conversions.

    This function orchestrates the pipeline by:
      1. Running frame analysis using the FramesAnalyzer with the provided keyword arguments (fa_kwargs).
      2. Processing the resulting frames directory to produce CSV outputs via the AnalysesProcessor using ap_kwargs.
      3. Converting the CSV outputs into matrices using the ToMatricesConverter with tmc_kwargs.
    
    After these steps, it safely deletes the intermediate directories (frames_dir and csv_dir) and returns the final output directory
    containing the network matrices.

    Args:
        fa (FramesAnalyzer): An instance of FramesAnalyzer.
        ap (AnalysesProcessor): An instance of AnalysesProcessor.
        tmc (ToMatricesConverter): An instance of ToMatricesConverter.
        fa_kwargs (dict): Keyword arguments for the FramesAnalyzer.analyse_frames() method.
        ap_kwargs (dict): Keyword arguments for the AnalysesProcessor.process_target_directory() method.
        tmc_kwargs (dict): Keyword arguments for the ToMatricesConverter.process_target_directory() method.

    Returns:
        str: The path to the output directory containing the network components.
    """
    frames_dir = fa.analyse_frames(**fa_kwargs)
    csv_dir = ap.process_target_directory(target_directory_path=frames_dir, **ap_kwargs)
    output_dir = tmc.process_target_directory(target_directory_path=csv_dir, **tmc_kwargs)

    # clean-up
    try:
        shutil.rmtree(frames_dir)
    except Exception as e:
        print(f"Warning: Could not delete frames directory '{frames_dir}'. Exception: {e}")
    try:
        shutil.rmtree(csv_dir)
    except Exception as e:
        print(f"Warning: Could not delete CSV directory '{csv_dir}'. Exception: {e}")

    return output_dir


__all__ = ["FramesAnalyzer", "AnalysesProcessor", "ToMatricesConverter", "Protein", "construct_network", "NetworkAnalyzer"]
