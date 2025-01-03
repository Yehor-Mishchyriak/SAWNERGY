#!/AllostericPathwayAnalyzer/venv/bin/python3

"""
This is a temporary main.py file that will be used while I am working on GUI
"""

import sys
import os
from core.Protein import Protein

def get_path_input(prompt, required=False, default=None):
    """Helper function to get a valid directory path from the user."""
    while True:
        user_input = input(prompt).strip()
        if user_input.lower() == "exit":
            sys.exit("Exiting. Goodbye!")
        
        if not user_input:
            if required:
                print("This field is required. Please enter a valid directory path.")
                continue
            else:
                return default
        
        if not os.path.exists(user_input):
            print("The specified directory does not exist. Please try again.")
            continue
        
        if not os.path.isdir(user_input):
            print("The specified path is not a directory. Please try again.")
            continue
        
        return user_input 

def get_int_input(prompt, required=False, default=None):
    """Helper function to get an integer input with validation."""
    while True:
        user_input = input(prompt)
        if user_input.lower() == "exit":
            sys.exit("Exiting. Goodbye!")
        if not user_input.strip() and not required:
            return default
        try:
            return int(user_input)
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

def get_float_input(prompt, required=False, default=None):
    """Helper function to get a float input with validation."""
    while True:
        user_input = input(prompt)
        if user_input.lower() == "exit":
            sys.exit("Exiting. Goodbye!")
        if not user_input.strip() and not required:
            return default
        try:
            return float(user_input)
        except ValueError:
            print("Invalid input. Please enter a valid float.")

def get_tuple_input(prompt, required=False, default=None):
    """Helper function to get a tuple input with validation."""
    while True:
        user_input = input(prompt)
        if user_input.lower() == "exit":
            sys.exit("Exiting. Goodbye!")
        if not user_input.strip() and not required:
            return default
        try:
            return tuple(map(int, user_input.split(",")))
        except ValueError:
            print("Invalid input. Please enter a tuple of integers separated by commas (e.g., 1,2,3).")

def get_bool_input(prompt, default=None):
    """Helper function to get a boolean input with validation."""
    while True:
        user_input = input(prompt + " (y/n): ").lower()
        if user_input == "exit":
            sys.exit("Exiting. Goodbye!")
        if not user_input.strip() and default is not None:
            return default
        if user_input in ["y", "yes"]:
            return True
        elif user_input in ["n", "no"]:
            return False
        else:
            print("Invalid input. Please enter 'y' for Yes or 'n' for No.")

def main():
    """Main function to handle user input."""
    
    print("Welcome! Please follow the prompts. To exit, enter 'exit' or press Ctrl+C.")
    network_directory_path = get_path_input("Input the network directory path (required): ")
    interactions_precision_limit_decimals = get_int_input("Input the interactions precision limit in decimals (optional, default=None): ", default=None)
    seed = get_int_input("Input the random seed (optional, default=None): ", default=None)

    with Protein(network_directory_path=network_directory_path,
                interactions_precision_limit_decimals=interactions_precision_limit_decimals,
                seed=seed) as p:
        try:
            while True:
                start_residue = get_int_input("Input the start residue: ", required=True)
                number_steps = get_int_input("Input the number of steps (optional, default=None): ", default=None)
                
                target_residues = get_tuple_input(
                    "Input the target residues as a comma-separated list (optional, default=None): ", default=None)
                
                number_pathways = get_int_input("Input the number of pathways (optional, default=None): ", default=None)
                
                filter_out_improbable = get_bool_input(
                    "Filter out improbable pathways? (optional, default=None): ", default=None)
                
                percentage_kept = get_float_input(
                    "Input the percentage of pathways to keep (optional, default=None): ", default=None)
                
                in_parallel = get_bool_input("Process in parallel? (optional, default=None): ", default=None)

                output_path = p.create_pathways(start_residue=start_residue,
                                    number_steps=number_steps,
                                    target_residues=target_residues,
                                    number_pathways=number_pathways,
                                    filter_out_improbable=filter_out_improbable,
                                    percentage_kept=percentage_kept,
                                    in_parallel=in_parallel)
                
                print(f"The pathways were successfully generated and saved to {output_path}")

        except KeyboardInterrupt:
            sys.exit("\nExiting. Goodbye!")
        except Exception as e:
            print(f"An error occurred: {e}. Please try again.")


if __name__ == "__main__":
    main()
