#!AllostericPathwayAnalyzer/venv/bin/python3

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Electrostatic network representation of the protein based on the input configuration file")
    parser.add_argument('topology_file', type=str, help='Path to the topology file')


if __name__ == "__main__":
    main()
