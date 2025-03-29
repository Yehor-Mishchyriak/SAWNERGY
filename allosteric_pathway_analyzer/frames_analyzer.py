# external imports
import os
from subprocess import run, SubprocessError
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

# local imports
from . import pkg_globals
from . import _util


class FramesAnalyzer:

    cpptraj_elec_analysis_command = "pairwise :{start_res_id}-{end_res_id} :{start_res_id}-{end_res_id} cuteelec {cutoff} avgout"
    cpptraj_vdw_analysis_command = "pairwise :{start_res_id}-{end_res_id} :{start_res_id}-{end_res_id} cutevdw {cutoff} avgout"
    cpptraj_hbond_analysis_command = "donormask :{start_res_id}-{end_res_id} acceptormask :{start_res_id}-{end_res_id} distance 3.5 angle 120 avgout"
    cpptraj_COM_analysis_command = "vector res_{res_id} center :{res_id} out"

    def __init__(self, cpptraj_abs_path: Optional[str] = None) -> None:
        self.global_config = None
        self.cls_config = None
        self.set_config(pkg_globals.default_config)

        self._cpptraj = _util.cpptraj_is_available_at(cpptraj_abs_path)
        if self._cpptraj is None:
            raise FileNotFoundError(f"cpptraj was not found or is inaccessible at {cpptraj_abs_path}")

    @property
    def which_cpptraj(self) -> Optional[str]:
        return self._cpptraj

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.cls_config}, _cpptraj={self._cpptraj})"

    def set_config(self, config: dict) -> None:
        if self.__class__.__name__ not in config:
            raise ValueError(f"Invalid config: missing {self.__class__.__name__} sub-config")
        
        sub_config = config[self.__class__.__name__]
        if "output_directory_name" not in sub_config:
            raise ValueError(f"Invalid {self.__class__.__name__} sub-config: missing output_directory_name field")
        if "cpptraj_file_name" not in sub_config:
            raise ValueError(f"Invalid {self.__class__.__name__} sub-config: missing cpptraj_file_name field")
        
        self.global_config = config
        self.cls_config = config[self.__class__.__name__]

    def run_cpptraj(self, start_end: Tuple[int, int], topology_file: str, trajectory_file: str,
                     cpptraj_analysis_command: str, cpptraj_output_type: str, output_directory: str) -> None:
        start_frame, end_frame = start_end
        try:
            output_file_name = self.cls_config["cpptraj_file_name"].format(start=start_frame, end=end_frame)
        except KeyError:
            raise KeyError(f"Wrong 'cpptraj_file_name' format. Expected a string containing {{\"start\"}}-{{\"end\"}}, instead got: {self.cls_config['cpptraj_file_name']}")
        output_file_path = os.path.join(output_directory, output_file_name)

        command = (
            f'echo "parm {topology_file}\n'
            f'trajin {trajectory_file} {start_frame} {end_frame}\n'
            f'{cpptraj_analysis_command} {output_file_path} run" | '
            f'{self._cpptraj} > /dev/null 2>&1'
        )
        try:
            run(command, check=True, shell=True)
        except SubprocessError as e:
            raise RuntimeError(f"An exception occurred while executing the '{command}' command: {e}")

    def analyse_frames(self, topology_file: str, trajectory_file: str,
                       cpptraj_analysis_command: str, cpptraj_output_type: str,
                       number_frames: int, in_parallel: bool, batch_size: Optional[int] = 1,
                       in_one_batch: Optional[bool] = False, output_directory_path: Optional[str] = None) -> str:
        output_directory = (
            output_directory_path if output_directory_path
            else _util.create_output_dir(os.getcwd(), self.cls_config["output_directory_name"])
        )

        # If in_one_batch is True, treat all frames as one batch.
        effective_batch_size = number_frames if in_one_batch else batch_size
        batches = _util.construct_batch_sequence(number_frames, effective_batch_size)
        
        _util.process_elementwise(
            in_parallel=in_parallel,
            Executor=ThreadPoolExecutor,
            capture_output=False
        )(
            batches,
            self.run_cpptraj,
            topology_file,
            trajectory_file,
            cpptraj_analysis_command,
            cpptraj_output_type,
            output_directory
        )
        
        return output_directory


if __name__ == "__main__":
    pass
