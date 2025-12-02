import yaml
from pathlib import Path
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import torch

def load_config(config_relative_path):
    current_file_dir = Path(__file__).parent 
    config_path = current_file_dir.parent / config_relative_path
    
    with open(config_path.resolve(), 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging(log_file_name):
    """Configure logging to log_file_name"""
    os.makedirs('logs', exist_ok=True)
    log_path = os.path.join('logs', log_file_name)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    def amsterdam_time(*args):
        return datetime.now(ZoneInfo("Europe/Amsterdam")).timetuple() # timezone amsterdam CEST instead of default UTC

    logging.Formatter.converter = amsterdam_time
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='a'),
            logging.StreamHandler() 
        ]
    )

    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    logging.info(f"Logging configured. Output added: {log_path}")

def get_device(config):
    device_cfg = config["training"]["device"]

    if device_cfg == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    elif device_cfg == "cpu":
        return torch.device("cpu")
    elif device_cfg == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        raise ValueError(f"Invalid device setting in config: {device_cfg}")
