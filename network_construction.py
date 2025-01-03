#!/AllostericPathwayAnalyzer/venv/bin/python3

from shutil import rmtree
from os import listdir

from core.FramesAnalyzer import FramesAnalyzer
from core.AnalysesProcessor import AnalysesProcessor
from core.ToMatricesConverter import ToMatricesConverter

# NOTE: of course one could make an interactive input for the parameters of FramesAnalyzer class
# HOWEVER, the computation is very expensive, so it is to be executed on an HPCC,
# so to make it simple for inclusion into a SLURM script, the parameters were hardcoded and are to be modified as necessary

fa = FramesAnalyzer(
    topology_file_path = "../p53_WT_nowat.prmtop",
    trajectory_file_path = "../p53_WT_md1000_str.nc",
    number_frames = 1000,
    cpptraj_analysis_command = "pairwise :* :* cuteelec 1.0",
    cpptraj_output_type = "avgout",
    start_frame = 1,
    batch_size = 10,
    in_one_batch = False,
    output_directory = None
)
ap = AnalysesProcessor()
tmc = ToMatricesConverter()

try:
    tmp_frames_dir = fa.analyse_frames() # analyze frames with cpptraj
    tmp_csvs_dir = ap.process_target_directory(tmp_frames_dir) # processes the analyses and save as .csv

    csv_files = listdir(tmp_csvs_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {tmp_csvs_dir}")
    csv_file = csv_files[0] # take a .csv file from the last step
    residues = tmc.map_id_to_res(csv_file) # and extract the residues map

    network = tmc.process_target_directory(tmp_csvs_dir, dimension=len(residues))
finally:
    rmtree(tmp_frames_dir); rmtree(tmp_csvs_dir)

print(f"The network files can be accessed at {network} directory")
