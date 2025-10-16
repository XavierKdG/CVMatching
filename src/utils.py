import logging
import os
import yaml
from datetime import datetime
from zoneinfo import ZoneInfo

def setup_logging(log_file_name='pipeline.log'):
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
    logging.info(f"Logging configured. Output added: {log_path}")

def load_config(config_path="configs/config.yml"):
    """Load configuration from a YAML file."""
    logging.info(f"Configuration loaded from: {config_path}")
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config

    
