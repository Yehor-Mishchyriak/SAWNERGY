# external imports
import os
from subprocess import run, SubprocessError
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

# local imports
from . import pkg_globals
from . import _util


class FramesAnalyzer:

    # * PROTECTED CLASS FIELDS *

    # purpose: extract pairwise electrostatic interactions between residues from a specific range
    # args: start residue, end residue, interaction energy cutoff value
    #       (if less than that, the interaction won't be recorded), path to the output file
    @staticmethod
    def _elec_analysis(start_res_id: int, end_res_id: int, cutoff: float, output_file_path: str):
        return f"pairwise :{start_res_id}-{end_res_id} :{start_res_id}-{end_res_id} cuteelec {cutoff} avgout {output_file_path}\nrun'"
    
    # purpose: extract pairwise van der Waals interactions between residues from a specific range
    # args: start residue, end residue, interaction energy cutoff value
    #       (if less than that, the interaction won't be recorded), path to the output file
    @staticmethod
    def _vdw_analysis(start_res_id: int, end_res_id: int, cutoff: float, output_file_path: str):
        return f"pairwise :{start_res_id}-{end_res_id} :{start_res_id}-{end_res_id} cutevdw {cutoff} avgout {output_file_path}\nrun'"

    # purpose: extract pairwise hydrogen bond interactions between residues from a specific range
    # args: start residue, end residue, distance cutoff value, angle cutoff value
    #       (if less than either of these, the interaction won't be recorded), path to the output file
    @staticmethod
    def _hbond_analysis(start_res_id: int, end_res_id: int, distance_cutoff: float, angle_cutoff: float, output_file_path: str):
        return (f"donormask :{start_res_id}-{end_res_id} acceptormask :{start_res_id}-{end_res_id} distance {distance_cutoff} angle {angle_cutoff} avgout {output_file_path}\nrun'")
    
    # purpose: extract the center of the mass of each individual residue from a specific residue range
    # args: start residue, end residue, template for the path to the output file
    @staticmethod
    def _com_analysis(start_res_id: int, end_res_id: int, output_file_path_template: str):
        s = ""
        for res_id in range(start_res_id, end_res_id+1):
            s += f"vector res_{res_id} center :{res_id} out {output_file_path_template.format(res_id=res_id)}\n"
        return s[:-1] + " run' "

    # * PUBLIC CLASS FIELDS *

    # purpose: load the molecular dynamics files
    # args: topology file, trajectory file, start frame, end frame
    @staticmethod
    def load_data_from(topology_file: str, trajectory_file: str, start_frame: int, end_frame: int):
        return (f"echo 'parm {topology_file}\ntrajin {trajectory_file} {start_frame} {end_frame}\n")
    
    # storing available analyses in the dict
    available_analyses = {"elec": _elec_analysis, "vdw": _vdw_analysis, "hbond": _hbond_analysis, "com": _com_analysis}

    def __init__(self, cpptraj_abs_path: Optional[str] = None, config: Optional[dict] = None) -> None:
        self.global_config = None
        self.cls_config = None
        
        if config is None:
            self.set_config(pkg_globals.default_config)
        else:
            self.set_config(config)

        self._cpptraj = _util.cpptraj_is_available_at(cpptraj_abs_path)
        if self._cpptraj is None:
            raise FileNotFoundError(f"cpptraj was not found or is inaccessible at {cpptraj_abs_path}")
        
        # just storing the piping operator and the cpptraj output suppresion so that
        # its usage in the instance methods does not make the code verbose
        self._through_cpptraj = f" | {self._cpptraj}" #> /dev/null 2>&1

    @property
    def which_cpptraj(self) -> Optional[str]:
        return self._cpptraj

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cls_config={self.cls_config}, _cpptraj={self._cpptraj})"

    def set_config(self, config: dict) -> None:
        self.global_config = config
        self.cls_config = config[self.__class__.__name__]

    def extract_residue_interactions(self, start_end_frames: Tuple[int, int], topology_file: str, trajectory_file: str,
                                     interaction_type: str, analysis_kwargs: dict, output_directory: str) -> None:
        
        start_frame, end_frame = start_end_frames
        output_file_name = self.cls_config["interactions_file_name_template"].format(start_frame=start_frame, end_frame=end_frame)
        output_file_path = os.path.join(output_directory, output_file_name)

        command = "".join([self.load_data_from(topology_file, trajectory_file, start_frame, end_frame),
                          self.interaction_analyses[interaction_type](**analysis_kwargs, output_file_path=output_file_path),
                          self._through_cpptraj])

        try:
            run(command, check=True, shell=True)
        except SubprocessError as e:
            raise RuntimeError(f"An exception occurred while executing the '{command}' command: {e}")

    def extract_residue_coordinates(self, start_end_residues: Tuple[int, int], start_end_frames: Tuple[int, int],
                                    topology_file: str, trajectory_file: str, output_directory: str):

        start_residue, end_residue = start_end_residues
        start_frame, end_frame = start_end_frames
        output_file_name_template = self.cls_config["coordinates_file_name_template"]
        output_file_path_template = os.path.join(output_directory, output_file_name_template)

        command = "".join([self.load_data_from(topology_file, trajectory_file, start_frame, end_frame),
                          self.com_analysis(start_residue, end_residue, output_file_path_template),
                          self._through_cpptraj])
        print(command)

        try:
            run(command, check=True, shell=True)
        except SubprocessError as e:
            raise RuntimeError(f"An exception occurred while executing the '{command}' command: {e}")

    def process_trajectory(self, topology_file: str, trajectory_file: str,
                       interaction_types_and_kwargs: dict, number_frames: int,
                       in_parallel: bool, batch_size: Optional[int] = 1,
                       in_one_batch: Optional[bool] = False, plotable: Optional[bool] = False, 
                       start_end_residues: Optional[Tuple[int, int]] = None, output_directory_path: Optional[str] = None) -> str:

        output_directory = (
            output_directory_path if output_directory_path
            else _util.create_output_dir(os.getcwd(), self.cls_config["output_directory_name_template"])
        )
        
        extract_interactions_from = _util.process_elementwise(in_parallel=in_parallel, Executor=ThreadPoolExecutor, capture_output=False)

        # If in_one_batch is True, treat all frames as one batch.
        batch_size = number_frames if in_one_batch else batch_size
        batches = _util.construct_batch_sequence(number_frames, batch_size)

        # extract interactions
        for interaction_type, analysis_kwargs in interaction_types_and_kwargs.items():
            #! ERROR: you need to actually create this directory: self.cls_config[f"{interaction_type}_directory_name"]!!!
            interaction_type_output_directory = os.path.join(output_directory, self.cls_config[f"{interaction_type}_directory_name"])
            extract_interactions_from(batches,
                            self.extract_residue_interactions,
                            topology_file, trajectory_file,
                            interaction_type, analysis_kwargs,
                            interaction_type_output_directory)

        if plotable:
            if start_end_residues is None:
                raise ValueError("If plotable=True, start_end_residues parameter must be provided")
            self.extract_residue_coordinates(start_end_residues=start_end_residues,
                                             start_end_frames=(1, number_frames),
                                             topology_file=topology_file,
                                             trajectory_file=trajectory_file,
                                             output_directory=output_directory)

        return output_directory


if __name__ == "__main__":
    pass
