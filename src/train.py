import os
import argparse
import pandas as pd
import shutil
import subprocess
import logging
import spacy
from spacy.cli.train import train as spacy_train
from spacy.cli.convert import convert as spacy_convert
from pathlib import Path
from .utils import setup_logging, load_config #help functions
import src.upload_to_qdrant as uploader

class ModelTrainer:
    """Handles the spaCy NER model training pipeline."""
    def __init__(self, config):
        """Initializes the trainer with paths from the config."""
        logging.info("Initializing ModelTrainer...")
        self.config = config
        self.spacy_config_path = Path(config["paths"]["spacy_config"])
        self.model_output_path = Path(config["paths"]["models_folder"])
        self.processed_dir = Path(config["paths"]["processed_folder"])
        
        self.train_spacy_file = self.processed_dir / "train.spacy"
        self.dev_spacy_file = self.processed_dir / "dev.spacy"

    def generate_spacy_config(self):
        """Generates the spacy_base_config.cfg file using settings from config.yml."""
        optimizer = self.config["training"]["optimizer"]
        seed = self.config["training"]["seed"]
        
        logging.info(f"Generating spaCy config at: {self.spacy_config_path}")
        
        command = [
            "python", "-m", "spacy", "init", "config",
            str(self.spacy_config_path),
            "--lang", "en",
            "--pipeline", "ner",
            "--optimize", optimizer,
            "--force"
        ]
        
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            
            config_text = self.spacy_config_path.read_text()
            config_text = config_text.replace("seed = 0", f"seed = {seed}")
            self.spacy_config_path.write_text(config_text)
            logging.info(f"Set seed to {seed} in {self.spacy_config_path}")
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to generate spaCy config: {e.stderr}")
            raise
        except Exception as e:
            logging.error(f"Failed to update seed in config file: {e}")
            raise

    def run_training(self):
        """Trains the spaCy NER model using the generated config and .spacy files. Streams the training output to the log."""
        logging.info("--- Starting spaCy Model Training ---")
        command = [
            "python", "-m", "spacy", "train", 
            str(self.spacy_config_path), 
            "--output", str(self.model_output_path), 
            "--paths.train", str(self.train_spacy_file), 
            "--paths.dev", str(self.dev_spacy_file)
            # "--gpu-id", '-1'  # Use GPU 0. Set to -1 to force CPU.
        ]
        
        logging.info(f"Running training command: {' '.join(command)}")
        
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    logging.info(line.strip())
            
            process.wait()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command, "Training process failed.")
                
            logging.info(f"--- Training Complete. Model saved to: {self.model_output_path} ---")
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Training failed: {e}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred during training: {e}")
            raise

    def load_model(self):
        """Loads the best trained model from the output directory."""
        best_model_path = self.model_output_path / "model-best"
        if not best_model_path.exists():
            logging.error(f"Trained model not found at {best_model_path}")
            return None
        
        logging.info(f"Loading trained model from {best_model_path}")
        return spacy.load(best_model_path)

    def run_pipeline(self):
        """Runs the full training pipeline: config + training."""
        self.generate_spacy_config()
        self.run_training()

def main(config_path=None):
    """Main function to run the training and Qdrant upload pipeline."""
    if config_path is None:
        config_path = "configs/config.yml" 
        
    config = load_config(config_path) 
    setup_logging(config["logging"]["file_name"])

    if config["pipeline"]["mode"] == "custom":
        logging.info("Pipeline mode is 'custom'. Proceeding with NER model training.")
        trainer = ModelTrainer(config)
        trainer.run_pipeline()
    else:
        logging.info("Pipeline mode is 'base'. Skipping NER model training.")

    logging.info("--- Proceeding to upload data to Qdrant ---")
    try:
        uploader.main(config_path=config_path)
    except Exception as e:
        logging.error(f"Failed to run Qdrant upload pipeline: {e}")
        raise
    
    logging.info("--- Full Train & Upload Pipeline Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yml", help="Path to the config.yml file")
    args = parser.parse_args()
    
    main(config_path=args.config)