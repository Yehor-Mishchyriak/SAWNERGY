#!AllostericPathwayAnalyzer/venv/bin/python3

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Electrostatic network representation of a protein based on the input configuration file")
    parser.add_argument('config_dir', type=str, help='Path to the configuration files directory')


if __name__ == "__main__":
    main()
