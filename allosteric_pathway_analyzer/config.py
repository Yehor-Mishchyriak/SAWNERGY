from json import load

def load_json_config(config_location: str) -> dict:
    """
    Loads a JSON configuration file.

    Args:
        config_location (str): Path to the JSON configuration file.

    Returns:
        dict: Parsed JSON configuration.
    """
    with open(config_location, "r") as config_file:
        config = load(config_file)
    return config
