# external imports
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

# local imports
from . import pkg_globals
from . import _util


class AnalysesProcessor:

    def __init__(self, config: Optional[dict] = None) -> None:
        self.global_config = None
        self.cls_config = None
        if config is None:
            self.set_config(pkg_globals.default_config)
        else:
            self.set_config(config)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.cls_config})"

    def set_config(self, config: dict) -> None:
        self.global_config = config
        self.cls_config = config[self.__class__.__name__]

    def _construct_csv_file_path(self, cpptraj_file_path: str, output_directory: str) -> Tuple[str, str]:
        cpptraj_file_name, analysis_type = _util.name_and_analysis_type_from_path(cpptraj_file_path)
        
        if analysis_type == "com":
            res_id = _util.residue_id_from_name(cpptraj_file_name)
            csv_file_name = self.cls_config["coordinates_file_name_template"].format(res_id=res_id)
        else:
            start_frame, end_frame = _util.frames_from_name(cpptraj_file_name)
            csv_file_name = self.cls_config["interactions_file_name_template"].format(start_frame=start_frame, end_frame=end_frame)

        output_directory = os.path.join(output_directory, analysis_type)
        csv_file_path = os.path.join(_util.new_dir_at(output_directory), csv_file_name)

        return csv_file_path, analysis_type

    def cpptraj_to_csv_immediately(self, cpptraj_file_path: str, output_directory: str) -> None:
        csv_file_path, analysis_type = self._construct_csv_file_path(cpptraj_file_path, output_directory)
        parser = _util.cpptraj_data_parsers[analysis_type]
        header = self.cls_config["com_csv_header" if analysis_type == "com" else "interactions_csv_header"]
        # read the entire cpptraj file
        contents = _util.read_lines(cpptraj_file_path)
        # write the CSV file with the provided header and contents
        _util.write_csv_from_cpptraj(parser, header, contents, csv_file_path)

    def cpptraj_to_csv_incrementally(self, cpptraj_file_path: str, output_directory: str,
                                     allowed_memory_percentage_hint: float, num_workers: int) -> None:
        csv_file_path, analysis_type = self._construct_csv_file_path(cpptraj_file_path, output_directory)
        parser = _util.cpptraj_data_parsers[analysis_type]
        header = self.cls_config["com_csv_header" if analysis_type == "com" else "interactions_csv_header"]
        # write the CSV header to the output file
        _util.write_csv_header(header, csv_file_path)
        # process the cpptraj file in chunks and append each chunk to the CSV file
        for chunk in _util.chunked_file(cpptraj_file_path, allowed_memory_percentage_hint, num_workers):
            _util.append_csv_from_cpptraj(parser, chunk, csv_file_path)

    def process_target_directory(self, target_directory_path: str,
                                in_parallel: bool, allowed_memory_percentage_hint: Optional[float] = None,
                                num_workers: Optional[int] = None, output_directory_path: Optional[str] = None) -> str:
        output_directory = output_directory_path if output_directory_path else _util.create_output_dir(os.getcwd(), self.cls_config["output_directory_name_template"])
        
        if in_parallel:
            if num_workers is None:
                raise ValueError("If in_parallel=True, num_workers parameter must be provided")
            if allowed_memory_percentage_hint is None:
                raise ValueError("If in_parallel=True, allowed_memory_percentage_hint parameter must be provided")
            convert_to_csv = _util.process_elementwise(in_parallel=True, Executor=ThreadPoolExecutor, capture_output=False, max_workers=num_workers)
        else:
            convert_to_csv = _util.process_elementwise(in_parallel=False, capture_output=False)

        target_subdirectory_paths = [os.path.join(target_directory_path, target_subdirectory) for target_subdirectory in os.listdir(target_directory_path)]        
        for path_ in target_subdirectory_paths:
            
            if not os.path.isdir(path_):
                if os.path.basename(path_) == self.cls_config["id_to_res_map_name"]:
                    os.rename(src=path_, dst=os.path.join(output_directory, self.cls_config["id_to_res_map_name"]))
                continue

            analysis_file_paths = [os.path.join(path_, file) for file in os.listdir(path_)]
            if in_parallel:
                convert_to_csv(analysis_file_paths, self.cpptraj_to_csv_incrementally, output_directory, allowed_memory_percentage_hint, num_workers)
            else:
                convert_to_csv(analysis_file_paths, self.cpptraj_to_csv_immediately, output_directory)

        return output_directory


if __name__ == "__main__":
    pass
