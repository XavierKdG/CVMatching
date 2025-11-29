import yaml
from pathlib import Path

def load_config(config_relative_path):
    current_file_dir = Path(__file__).parent 
    config_path = current_file_dir.parent / config_relative_path
    
    with open(config_path.resolve(), 'r') as file:
        config = yaml.safe_load(file)
    return config