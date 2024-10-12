#!AllostericPathwayAnalyzer/venv/bin/python3

import os
import util
from subprocess import run
from concurrent.futures import ThreadPoolExecutor

root_config = util.load_json_config("AllostericPathwayAnalyzer/configs/root.json")
logger = util.set_up_logging("AllostericPathwayAnalyzer/configs/logging.json", "network_construction_module")
class FramesAnalyzer:

    def __init__(self, topology_file: str, trajectory_file: str,
                 number_frames: int,
                 cpptraj_analysis_command: str, cpptraj_output_type: str,
                 batch_size: int = 1, in_one_batch: bool = False) -> None:

        # MD Simulation data import
        self.topology_file = topology_file
        self.trajectory_file = trajectory_file

        # cpptraj parameters
        self.cpptraj_analysis_command = cpptraj_analysis_command
        self.cpptraj_output_type = cpptraj_output_type

        # frame batches
        batch_size = number_frames if in_one_batch else batch_size
        self.batches = util.construct_batch_sequence(number_frames, batch_size)
        # cpptraj output directory
        self.output_directory = util.create_output_dir(root_config["GLOBAL"]["output_directory_path"], root_config["FramesAnalyzer"]["output_directory_name"])

    def _run_cpptraj(self, start_end: tuple) -> None:
        start_frame, end_frame = start_end
        output_file_path = os.path.join(self.output_directory, f"{start_frame}-{end_frame}.dat")

        command = f"""echo \"parm {self.topology_file}
                    trajin {self.trajectory_file} {start_frame} {end_frame}
                    {self.cpptraj_analysis_command} {self.cpptraj_output_type} {output_file_path} run\" | cpptraj > /dev/null 2>&1"""

        run(command, check=True, shell=True)

    def analyse_frames(self) -> str:
        if __name__ == "__main__":
            util.process_elementwise(True, ThreadPoolExecutor)(self.batches, self._run_cpptraj)
        else:
            util.process_elementwise(False, ThreadPoolExecutor)(self.batches, self._run_cpptraj)
        
        return self.output_directory


def main():
    pass


if __name__ == "__main__":
    main()
