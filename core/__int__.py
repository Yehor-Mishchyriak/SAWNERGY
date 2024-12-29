from .util import load_json_config, set_up_logging, create_output_dir
from os import getenv

root_config: dict = load_json_config(getenv("ROOT_CONFIG_PATH"))
network_construction_logger: dict = set_up_logging(getenv("LOGGING_CONFIG_PATH"), "network_construction_module", "queue_handler_construction_module")
protein_module_logger: dict = set_up_logging(getenv("LOGGING_CONFIG_PATH"), "protein_module", "queue_handler_protein_module")
