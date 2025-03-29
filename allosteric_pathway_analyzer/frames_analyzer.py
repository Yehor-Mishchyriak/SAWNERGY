# external imports
import os
from subprocess import run, SubprocessError
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

# local imports
from . import pkg_globals
from . import _util


class FramesAnalyzer:

    _from_md = lambda topology_file, trajectory_file, start_frame, end_frame: (
                    f"echo 'parm {topology_file}\n"
                    f"trajin {trajectory_file} {start_frame} {end_frame}\n")

    _elec_analysis = lambda start_res_id, end_res_id, cutoff, output_file_path: (
                    f"pairwise :{start_res_id}-{end_res_id}"
                    f":{start_res_id}-{end_res_id}"
                    f"cuteelec {cutoff} avgout {output_file_path} run'")
    
    _vdw_analysis = lambda start_res_id, end_res_id, cutoff, output_file_path: (
                    f"pairwise :{start_res_id}-{end_res_id}"
                    f":{start_res_id}-{end_res_id}"
                    f"cutevdw {cutoff} avgout {output_file_path} run'")

    _hbond_analysis = lambda start_res_id, end_res_id, distance_cutoff, angle_cutoff, output_file_path: (
                    f"donormask :{start_res_id}-{end_res_id}"
                    f"acceptormask :{start_res_id}-{end_res_id}"
                    f"distance {distance_cutoff} angle {angle_cutoff} avgout {output_file_path} run'")

    interaction_analyses = {"elec": _elec_analysis, "vdw": _vdw_analysis, "hbond": _hbond_analysis}
    com_analysis = lambda start_res_id, end_res_id, output_file_path: FramesAnalyzer.__com_analysis_commands(start_res_id, end_res_id, output_file_path)

    @staticmethod
    def __com_analysis_commands(start_res_id: int, end_res_id: int, output_file_path: str):
        s = ""
        for res_id in range(start_res_id, end_res_id+1):
            s += f"vector res_{res_id} center :{res_id} out {output_file_path.format(res_id=res_id)}\n"
        return s[:-1] + " run' "

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
        output_file_name = self.cls_config["interactions_file_name"].format(start_frame=start_frame, end_frame=end_frame)
        output_file_path = os.path.join(output_directory, output_file_name)

        command = "".join([self._from_md(topology_file, trajectory_file, start_frame, end_frame),
                          self.interaction_analyses[interaction_type](**analysis_kwargs, output_file_path),
                          f"| {self._cpptraj} > /dev/null 2>&1"])

        try:
            run(command, check=True, shell=True)
        except SubprocessError as e:
            raise RuntimeError(f"An exception occurred while executing the '{command}' command: {e}")

    def extract_residue_coordinates(self, start_end_frames: Tuple[int, int], start_end_residues: Tuple[int, int], topology_file: str, trajectory_file: str,
                                    output_directory: str):
        start_frame, end_frame = start_end_frames
        start_residue, end_residue = start_end_residues
        output_file_name = self.cls_config["coordinates_file_name"]
        output_file_path = os.path.join(output_directory, output_file_name)

        command = "".join([self._from_md(topology_file, trajectory_file, start_frame, end_frame),
                          self.com_analysis(start_residue, end_residue, output_file_path),
                          f"| {self._cpptraj} > /dev/null 2>&1"])

        try:
            run(command, check=True, shell=True)
        except SubprocessError as e:
            raise RuntimeError(f"An exception occurred while executing the '{command}' command: {e}")

    def process_trajectory(self, topology_file: str, trajectory_file: str,
                       interaction_types_n_kwargs: dict, number_frames: int,
                       in_parallel: bool, batch_size: Optional[int] = 1,
                       in_one_batch: Optional[bool] = False, plotable: Optional[bool] = False, 
                       start_end_residues: Tuple[int, int] = None, output_directory_path: Optional[str] = None) -> str:
        output_directory = (
            output_directory_path if output_directory_path
            else _util.create_output_dir(os.getcwd(), self.cls_config["output_directory_name"])
        )

        # If in_one_batch is True, treat all frames as one batch.
        effective_batch_size = number_frames if in_one_batch else batch_size
        batches = _util.construct_batch_sequence(number_frames, effective_batch_size)
        
        process_frames = _util.process_elementwise(in_parallel=in_parallel,
                                            Executor=ThreadPoolExecutor,
                                            capture_output=False)
        
        # extract interactions
        for interaction_type, analysis_kwargs in interaction_types_n_kwargs:
            interaction_type_output_directory = os.path.join(output_directory, self.cls_config[f"{interaction_type}_directory_name"])
            process_frames(batches,
                            self.extract_residue_interactions,
                            topology_file, trajectory_file,
                            interaction_type, analysis_kwargs,
                            interaction_type_output_directory)

        if plotable:
            if start_end_residues is None:
                raise ValueError("If plotable=True, start_end_residues parameter must be provided")
            process_frames(...) # I stopped here for now; Need to alter the code a bit, so that
                                # it's required to provide an iterable of residues to compute COMs for



        return output_directory


if __name__ == "__main__":
    pass
