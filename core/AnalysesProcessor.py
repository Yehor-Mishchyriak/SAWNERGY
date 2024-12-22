#!AllostericPathwayAnalyzer/venv/bin/python3

import os
from concurrent.futures import ThreadPoolExecutor

# package imports
import core
from util import read_lines, write_csv, process_elementwise

class AnalysesProcessor:

    def __init__(self) -> None:
        self.output_directory = core.create_output_dir(core.root_config["GLOBAL"]["output_directory_path"],
                                                       core.root_config["AnalysesProcessor"]["output_directory_name"])
        core.network_construction_logger.info(f"Successfully created the output directory for {self.__class__.__name__} class.")

    @staticmethod
    def _cpptraj_to_csv(cpptraj_file_path: str, output_file_path: str) -> None:
        contents = read_lines(cpptraj_file_path)
        write_csv(contents, output_file_path)

    def process_target_directory(self, target_directory_path: str) -> str:

        analysis_files_paths = (os.path.join(target_directory_path, file) for file in os.listdir(target_directory_path))

        if __name__ == "__main__":
            process_elementwise(in_parallel=True, Executor=ThreadPoolExecutor)(analysis_files_paths, self._cpptraj_to_csv)
        else:
            process_elementwise(in_parallel=False)(analysis_files_paths, self._cpptraj_to_csv)

        return self.output_directory

def main():
    pass


if __name__ == "__main__":
    main()
