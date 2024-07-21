# still working on it
from subprocess import run, CalledProcessError
from Protein import Protein

configuration_pipeline = "configure_network.sh"
# need to get .pdb file dict somehow
def main():
    network_parameters = run(configuration_pipeline)
    protein = Protein()

if __name__ == "__main__":
    main()
