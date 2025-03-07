import os
from .config import load_json_config

_default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_config.json")

try:
    default_config: dict = load_json_config(_default_config_path)
except FileNotFoundError:
    raise RuntimeError(f"Configuration file 'default_config.json' not found at {_default_config_path}.")

__all__ = ["default_config"]


if __name__ == "__main__":
    pass
