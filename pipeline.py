import logging
import traceback
import argparse
from src.preprocess import main as run_preprocessing
from src.train import main as run_spacy_pipeline 
from src.utils import setup_logging, load_config

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the full CV Matching data pipeline.")
    parser.add_argument("--config", type=str, default="configs/config.yml", help="Path to the configuration YAML file.")
    return parser.parse_args()

def main():
    """Executes the complete pipeline"""
    args = parse_arguments()
    
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"CRITICAL: Failed to load configuration file {args.config}. Error: {e}")
        return 

    setup_logging(config["logging"]["file_name"])

    logging.info("==============================================")
    logging.info("====== STARTING THE FULL PIPELINE RUN ======")
    logging.info(f"Using config file: {args.config}")
    logging.info(f"Pipeline Mode: {config['pipeline']['mode']}")
    logging.info("==============================================")

    try:
        logging.info("--- [STEP 1/2] Kicking off data preprocessing... ---")
        run_preprocessing(config_path=args.config) 
        logging.info("--- [STEP 1/2] Data preprocessing finished successfully. ---")
 
        logging.info("--- [STEP 2/2] Starting SpaCy pipeline (train/upload)... ---")
        run_spacy_pipeline(config_path=args.config)
        logging.info("--- [STEP 2/2] SpaCy pipeline finished successfully. ---")

        logging.info("===================================================")
        logging.info("====== ✅ PIPELINE COMPLETED SUCCESSFULLY ✅ ======")
        logging.info("===================================================")

    except FileNotFoundError as fnf_error:
        logging.error(f"Pipeline failed due to a missing file: {fnf_error}")
        logging.error("Ensure previous steps (like preprocessing) completed successfully and files exist.")
        logging.error(f"Traceback:\n{traceback.format_exc()}")
        logging.critical("==========================================")
        logging.critical("====== ❌ PIPELINE FAILED TO COMPLETE ❌ ======")
        logging.critical("==========================================")
    except Exception as e:
        logging.error(f"Pipeline execution failed with an unexpected error: {e}")
        logging.error(f"Traceback:\n{traceback.format_exc()}")
        logging.critical("==========================================")
        logging.critical("====== ❌ PIPELINE FAILED TO COMPLETE ❌ ======")
        logging.critical("==========================================")

if __name__ == "__main__":
    main()