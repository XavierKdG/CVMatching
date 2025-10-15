import logging
import traceback
from src.preprocess import main as run_preprocessing
from src.train import main as run_training
from src.upload_to_qdrant import main as run_upload
from src.utils import setup_logging

def main():
    """
    Executes the complete data processing, model training, and database upload pipeline.
    """
    setup_logging(log_file_name='full_pipeline.log')
    logging.info("==============================================")
    logging.info("====== STARTING THE FULL PIPELINE RUN ======")
    logging.info("==============================================")

    try:
        logging.info("--- [STEP 1/3] Kicking off data preprocessing... ---")
        run_preprocessing()
        logging.info("--- [STEP 1/3] Data preprocessing finished successfully. ---")
  
        logging.info("--- [STEP 2/3] Starting model training and embedding generation... ---")
        run_training()
        logging.info("--- [STEP 2/3] Model training completed successfully. ---")

        logging.info("--- [STEP 3/3] Starting upload of embeddings to Qdrant... ---")
        run_upload()
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