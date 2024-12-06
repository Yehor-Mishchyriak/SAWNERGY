#!AllostericPathwayAnalyzer/venv/bin/python3

# external imports
import os
from subprocess import run, CalledProcessError
from concurrent.futures import ThreadPoolExecutor

# package imports
import core
from util import construct_batch_sequence, process_elementwise


class FramesAnalyzer:

    def __init__(self, topology_file: str, trajectory_file: str,
                 number_frames: int,
                 cpptraj_analysis_command: str, cpptraj_output_type: str,
                 batch_size: int = 1, in_one_batch: bool = False) -> None:
        try:
            # MD simulation data import
            self.topology_file = topology_file
            self.trajectory_file = trajectory_file

            # cpptraj parameters
            self.cpptraj_analysis_command = cpptraj_analysis_command
            self.cpptraj_output_type = cpptraj_output_type

            # frame batches
            batch_size = number_frames if in_one_batch else batch_size
            self.batches = construct_batch_sequence(number_frames, batch_size)
            # cpptraj output directory
            self.output_directory = core.create_output_dir(core.root_config["GLOBAL"]["output_directory_path"], core.root_config["FramesAnalyzer"]["output_directory_name"])

            core.network_construction_logger.info(f"Successfully initialized {self.__class__.__name__} class.")

        except KeyError as e:
            core.network_construction_logger.error(f"Missing config key: {e}")
            raise

        except Exception as e:
            core.network_construction_logger.error(f"Unexpected error occurred during {self.__class__.__name__} initialisation: {e}")
            raise

    def _run_cpptraj(self, start_end: tuple) -> None:
        start_frame, end_frame = start_end
        output_file_path = os.path.join(self.output_directory, core.root_config["FramesAnalyzer"]["start_end_frames_dependent_analysis_file_name"].format(start_frame, end_frame))

        command = f"""echo \"parm {self.topology_file}
                    trajin {self.trajectory_file} {start_frame} {end_frame}
                    {self.cpptraj_analysis_command} {self.cpptraj_output_type} {output_file_path} run\" | cpptraj > /dev/null 2>&1"""
        try:
            core.network_construction_logger.info(f"Began processing {start_frame}-{end_frame} frame(s) in _run_cpptraj function.")
            run(command, check=True, shell=True)
            core.network_construction_logger.info(f"Successfully processed the frame(s).")

        except CalledProcessError as e:
            core.network_construction_logger.error(f"cpptraj invoked by _run_cpptraj failed for frame(s) {start_frame}-{end_frame}: {e}")
            raise
        except Exception as e:
            core.network_construction_logger.error(f"Unexpected error occurred during _run_cpptraj execution for {start_frame}-{end_frame} frame(s): {e}")
            raise

    def analyse_frames(self) -> str:
        try:
            if __name__ == "__main__":
                core.network_construction_logger.info(f"Began processing frame(s) in parallel.")
                process_elementwise(in_parallel=True, Executor=ThreadPoolExecutor)(self.batches, self._run_cpptraj)
            else:
                core.network_construction_logger.info(f"Began processing frame(s) sequentially.")
                process_elementwise(in_parallel=False)(self.batches, self._run_cpptraj)

            core.network_construction_logger.info(f"Successfully processed all the frame(s).")
            return self.output_directory

        except Exception as e:
            core.network_construction_logger.error(f"Unexpected error occurred during analyse_frames execution: {e}")


def main():
    """To be filled in"""
    pass


if __name__ == "__main__":
    main()
