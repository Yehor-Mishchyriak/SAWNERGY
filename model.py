#!/AllostericPathwayAnalyzer/venv/bin/python3

import argparse
import sys
import os
from time import time

# Add the directory containing core to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
import core.newProtein

def main():
    parser = argparse.ArgumentParser(description="Create an electrostatic network representation of a protein")
    parser.add_argument('config_dir', type=str, help='Path to the configuration files directory')
    parser.add_argument('--interactions_precision_limit', type=int, default=1, help='Precision limit for interactions')
    parser.add_argument('--random_seed', type=int, help='Seed for random number generation')

    args = parser.parse_args()

    protein = core.newProtein.Protein(args.config_dir, args.interactions_precision_limit, args.random_seed)
    print(f"Protein initialized with {protein.number_residues} residues and {protein.number_matrices} matrices.")

    try:
        t1 = time()
        output_file = protein.create_pathways(
            start_residue=168,
            number_steps=300,
            target_residues=(67,),
            number_pathways=1,
            filter_out_improbable=False,
            percentage_kept=0.1)
        print(f"Pathways generation took {time() - t1:.3f} second(s)")
        print(f"They have been saved to: {output_file}")
    finally:
        protein.memory_cleanup()


if __name__ == "__main__":
    main()
