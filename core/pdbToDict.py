#!AllostericPathwayAnalyzer/venv/bin/python3

from FramesAnalyzer import FramesAnalyzer


def main():

    import argparse

    parser = argparse.ArgumentParser(description="Convert .pdb file into a python dictionary")
    parser.add_argument("pdb_file", type=str, help="Path to the pdb file")
    parser.add_argument("output_directory", type=str, default=None, help="Directory to save the dictionary to")
    
    args = parser.parse_args()
    
    # save the indices to residues map of the protein to the "output_directory" passed as an argument
    FramesAnalyzer.extract_residues_from_pdb(args.pdb_file, save_output=True, output_directory=args.output_directory)


if __name__ == "__main__":
    main()
