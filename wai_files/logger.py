# utils/logger.py
import logging
import os

def setup_logger(name="default", log_file="./logs/main.log"):
    """
    Sets up a logger that logs to both console and a single file.
    
    Args:
        name: Logger name
        log_file: Fixed log file path to store all logs
    
    Returns:
        logger object
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger  # Avoid adding handlers multiple times

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler (append mode)
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"Logger initialized. Writing to {log_file}")
    return logger
