from typing import Union, Any

from .frames_analyzer import FramesAnalyzer
from .analyses_processor import AnalysesProcessor
from .to_matrices_converter import ToMatricesConverter

# def pipeline(fa: Union[None, FramesAnalyzer] = None, fa_analyse_frames_kwargs: Union[None, dict] = None,
#              ap: Union[None, AnalysesProcessor] = None, ap_process_target_directory_kwargs: Union[None, dict] = None,
#              tmc: Union[None, Any] = None, tmc_process_target_directory_kwargs: Union[None, dict] = None) -> Union[None, str]:
#     # Note ^ tmc is either None or ToMatricesConverter; the reason it's Any now is because I haven't imported tmc yet

#     # VALIDATE THE DICTIONARIES
#         # HERE

#     result = None

#     if fa is not None:
#         result = fa_output_dir = fa.analyse_frames(**fa_analyse_frames_kwargs)
#         ap_process_target_directory_kwargs["target_directory_path"] = fa_output_dir

#     if ap is not None:
#         result = ap_output_dir = ap.process_target_directory(**ap_process_target_directory_kwargs)
#         tmc_process_target_directory_kwargs["target_directory_path"] = ap_output_dir

#     if tmc is not None:
#         result = tmc.process_target_directory(**tmc_process_target_directory_kwargs)

#     # CLEAN UP fa_output_dir and ap_output_dir; Do try-except-finally

#     return result

__all__ = ["FramesAnalyzer", "AnalysesProcessor", "ToMatricesConverter", "Protein", "pipeline"]
