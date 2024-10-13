#!/AllostericPathwayAnalyzer/venv/bin/python3

import argparse
import sys
import os

# Add the directory containing core to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
from core.Protein import Protein

def interactive_create_pathways(protein):
    """
    Continuously ask for parameters to create pathways and invoke the create_pathways method.
    """
    print("Enter 'exit' at any prompt to quit.")
    while True:
        try:
            perturbed_residue = input("Enter the perturbed residue index: ")
            if perturbed_residue.lower() == 'exit':
                break
            # account for zero-indexing
            perturbed_residue = int(perturbed_residue) - 1
            
            number_iterations = input("Enter the number of iterations (leave blank for default): ")
            if number_iterations.lower() == 'exit':
                break
            # account for zero-indexing
            number_iterations = int(number_iterations)-1 if number_iterations else None
            
            target_residues_input = input("Enter the target residues as comma-separated values (leave blank for none): ")
            if target_residues_input.lower() == 'exit':
                break
            if target_residues_input:
                # account for zero-indexing
                target_residues = set(map(lambda x: int(x)-1, target_residues_input.split(',')))
            else:
                target_residues = None
            
            number_pathways = input("Enter the number of pathways to generate: ")
            if number_pathways.lower() == 'exit':
                break
            number_pathways = int(number_pathways)
            
            filter_out_improbable = input("Filter out improbable pathways? (yes/no): ")
            if filter_out_improbable.lower() == 'exit':
                break
            filter_out_improbable = filter_out_improbable.strip().lower() == 'yes'
            
            if filter_out_improbable:
                percentage_kept = input("Enter the percentage of pathways to keep (as a decimal): ")
                if percentage_kept.lower() == 'exit':
                    break
                percentage_kept = float(percentage_kept)
            else:
                percentage_kept = 0.1
            
            output_directory = input("Enter the output directory (leave blank for current directory): ")
            if output_directory.lower() == 'exit':
                break
            output_directory = output_directory if output_directory else None
            
            output_dir = protein.create_pathways(
                perturbed_residue=perturbed_residue,
                number_iterations=number_iterations,
                target_residues=target_residues,
                number_pathways=number_pathways,
                filter_out_improbable=filter_out_improbable,
                percentage_kept=percentage_kept,
                output_directory=output_directory
            )
            print(f"Pathways have been saved to: {output_dir}")

        except ValueError:
            print("Invalid input, please enter the correct type of value.")
        except Exception as e:
            print(f"An error occurred: {e}")
        except KeyboardInterrupt:
            print("\nExiting interactive pathway creation.")
            break


def main():
    parser = argparse.ArgumentParser(description="Create an electrostatic network representation of a protein")
    parser.add_argument('config_dir', type=str, help='Path to the configuration files directory')
    parser.add_argument('--interactions_precision_limit', type=int, default=1, help='Precision limit for interactions')
    parser.add_argument('--random_seed', type=int, help='Seed for random number generation')

    args = parser.parse_args()

    try:
        protein = Protein(config_directory_path=args.config_dir, 
                          interactions_precision_limit=args.interactions_precision_limit, 
                          random_seed=args.random_seed)
        print(f"Protein initialized with {protein.number_residues} residues and {protein.number_matrices} matrices.")
        interactive_create_pathways(protein)
    except Exception as e:
        print(f"Failed to initialize Protein: {e}")


if __name__ == "__main__":
    main()
