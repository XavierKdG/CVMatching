import logging
import traceback
import argparse
from src.preprocess import main as run_preprocessing
from src.train import main as run_training
from src.upload_to_qdrant import main as run_upload
from src.utils import setup_logging

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the full data pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yml",
        help="Path to the configuration YAML file."
    )
    return parser.parse_args()

def main():
    """
    Executes the complete data processing, model training, and database upload pipeline.
    """
    args = parse_arguments()
    setup_logging(log_file_name='full_pipeline.log')

    logging.info("==============================================")
    logging.info("====== STARTING THE FULL PIPELINE RUN ======")
    logging.info(f"Using config file: {args.config}")
    logging.info("==============================================")

    try:
        logging.info("--- [STEP 1/3] Kicking off data preprocessing... ---")
        run_preprocessing(config_path=args.config)
        logging.info("--- [STEP 1/3] Data preprocessing finished successfully. ---")
  
        logging.info("--- [STEP 2/3] Starting model training and embedding generation... ---")
        run_training(config_path=args.config)
        logging.info("--- [STEP 2/3] Model training completed successfully. ---")

        logging.info("--- [STEP 3/3] Starting upload of embeddings to Qdrant... ---")
        run_upload(config_path=args.config)
        logging.info("--- [STEP 3/3] Upload to Qdrant completed successfully. ---")

        logging.info("===================================================")
        logging.info("====== ✅ PIPELINE COMPLETED SUCCESSFULLY ✅ ======")
        logging.info("===================================================")

    except Exception as e:
        logging.error(f"Pipeline execution failed with an error: {e}")
        logging.error(f"Traceback:\n{traceback.format_exc()}")
        logging.critical("==========================================")
        logging.critical("====== ❌ PIPELINE FAILED TO COMPLETE ❌ ======")
        logging.critical("==========================================")


if __name__ == "__main__":
    main()