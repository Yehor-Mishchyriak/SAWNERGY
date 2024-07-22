#!/AllostericPathwayAnalyzer/venv/bin/python3

from subprocess import run, CalledProcessError
from shutil import rmtree
import argparse

def run_command(command):
    result = run(command, capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        print(f"Command: {' '.join(command)}")
        print(f"Exit code: {result.returncode}")
        print(f"Standard output: {result.stdout}")
        print(f"Standard error: {result.stderr}")
        raise CalledProcessError(result.returncode, command, result.stdout, result.stderr)
    return result.stdout.strip()

def analyze_frames(topology_file, trajectory_file, number_frames, cpptraj_analysis_command, cpptraj_output_type, start_frame, batch_size, in_one_batch, output_directory):
    command = [
        "python3", "core/FramesAnalyzer.py",
        topology_file, trajectory_file, str(number_frames),
        cpptraj_analysis_command, cpptraj_output_type,
        "--start_frame", str(start_frame),
        "--batch_size", str(batch_size),
        "--output_directory", output_directory
    ]
    if in_one_batch:
        command.append("--in_one_batch")
    return run_command(command)

def process_analyses(target, output_directory):
    command = [
        "python3", "core/AnalysesProcessor.py",
        target, "--output_directory", output_directory
    ]
    return run_command(command)

def convert_atoms_to_residues(target, output_directory):
    command = [
        "python3", "core/AtToResConverter.py",
        target, "--output_directory", output_directory
    ]
    return run_command(command)

def convert_interactions_to_probs(target, output_directory):
    command = [
        "python3", "core/InteractionsToProbsConverter.py",
        target, "--output_directory", output_directory
    ]
    return run_command(command)

def convert_pdb_to_dict(pdb_file, output_directory):
    command = [
        "python3", "core/pdbToDict.py",
        pdb_file, output_directory
    ]
    return run_command(command)

def clean_up(dirs):
    for dir in dirs:
        rmtree(dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="Based on the provided topology, trajectory, .pdb file, and some additional parameters, generate the configuration file for the electrostatic network representation of the protein interpretable by model.py")
    parser.add_argument('topology_file', type=str, help='Path to the topology file')
    parser.add_argument('trajectory_file', type=str, help='Path to the trajectory file')
    parser.add_argument('pdb_file', type=str, help='Path to the .pdb file')
    parser.add_argument('number_frames', type=int, help='Total number of frames')
    parser.add_argument('cpptraj_analysis_command', type=str, help='cpptraj analysis command')
    parser.add_argument('cpptraj_output_type', type=str, help='cpptraj output type')
    parser.add_argument('--start_frame', type=int, default=1, help='The starting frame (default: 1)')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size (default: 1)')
    parser.add_argument('--in_one_batch', action='store_true', help='Process all frames in one batch')
    parser.add_argument('--output_directory', type=str, default=None, help='Directory to save the output files to')

    args = parser.parse_args()

    try:
        target1 = analyze_frames(args.topology_file, args.trajectory_file, args.number_frames, args.cpptraj_analysis_command, args.cpptraj_output_type, args.start_frame, args.batch_size, args.in_one_batch, args.output_directory)
        target2 = process_analyses(target1, args.output_directory)
        target3 = convert_atoms_to_residues(target2, args.output_directory)
        final = convert_interactions_to_probs(target3, args.output_directory)
        convert_pdb_to_dict(args.pdb_file, final)
        clean_up([target1, target2, target3])

        print("Configuration files were generated successfully! The following is the path to the config directory:")
        print("*NOTE: DO NOT MODIFY THIS DIRECTORY*")
        print(final)

    except CalledProcessError as e:
        print(f"Error occurred while executing: {e.cmd}")
        print(f"Exit code: {e.returncode}")
        print(f"Standard output: {e.stdout}")
        print(f"Standard error: {e.stderr}")

if __name__ == "__main__":
    main()
