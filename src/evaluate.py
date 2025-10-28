import spacy
import pandas as pd
import argparse
import logging
from pathlib import Path
import json
from .utils import setup_logging, load_config

def load_ner_model(config: dict):
    """
    Loads the correct NER model based on the pipeline mode in the config.
    
    Args:
        config: The loaded configuration dictionary.
    
    Returns:
        A loaded spaCy nlp object, or None if loading fails.
    """
    pipeline_mode = config.get("pipeline", {}).get("mode", "base")
    
    if pipeline_mode == "custom":
        model_path = Path(config["paths"]["models_folder"]) / "model-best"
        logging.info(f"Loading CUSTOM model from: {model_path}")
        if not model_path.exists():
            logging.error(f"Custom model path not found: {model_path}")
            logging.error("Please run the training pipeline first (src/train.py).")
            return None
        try:
            nlp = spacy.load(model_path)
            return nlp
        except Exception as e:
            logging.error(f"Failed to load custom model: {e}")
            return None
    else:
        model_name = config["models"]["base_model"]
        logging.info(f"Loading BASE model: {model_name}")
        try:
            nlp = spacy.load(model_name)
            return nlp
        except IOError:
            logging.error(f"Failed to load base model '{model_name}'.")
            logging.error(f"Try running: python -m spacy download {model_name}")
            return None
        except Exception as e:
            logging.error(f"An error occurred loading base model: {e}")
            return None

def evaluate_dataset(nlp, parquet_path: Path, output_path: Path, text_column: str = "cleaned_text"):
    """
    Runs the nlp model over a parquet file and saves the found entities 
    as a .jsonl "silver" annotation file.
    
    Args:
        nlp: The loaded spaCy model.
        parquet_path: Path to the input .parquet file.
        output_path: Path to save the resulting .jsonl file.
        text_column: The name of the column containing the text to process.
    """
    if not parquet_path.exists():
        logging.warning(f"Input file not found, skipping: {parquet_path}")
        return

    logging.info(f"Processing {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    if text_column not in df.columns:
        logging.error(f"'{text_column}' not found in {parquet_path}. Skipping.")
        return

    texts = df[text_column].astype(str).tolist()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_ents_found = 0
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for doc in nlp.pipe(texts, batch_size=50):
            entities = [[ent.start_char, ent.end_char, ent.label_] for ent in doc.ents]
            
            if entities:
                total_ents_found += len(entities)
    
            record = {
                "text": doc.text,
                "label": entities 
            }
            f_out.write(json.dumps(record) + "\n")

    logging.info(f"Finished processing. Found {total_ents_found} entities.")
    logging.info(f"Silver annotations saved to: {output_path}")

def main(config_path: str):
    """
    Main function to run the evaluation pipeline.
    """
    config = load_config(config_path)
    setup_logging(config["logging"]["file_name"])
    logging.info("--- Starting NER Silver Annotation Pipeline ---")

    nlp = load_ner_model(config)
    if nlp is None:
        logging.error("Failed to load NER model. Aborting.")
        return

    logging.info(f"Using model: {nlp.meta['name']} ({nlp.meta['version']})")
    
    processed_dir = Path(config["paths"]["processed_folder"])
    silver_dir = Path(config["paths"]["silver_folder"])

    try:
        jobs_cfg = config["datasets"]["jobs"]
        jobs_input_stem = Path(jobs_cfg["input_filename"]).stem
        jobs_parquet_path = processed_dir / f"{jobs_input_stem}_processed.parquet"
        jobs_silver_path = silver_dir / f"{jobs_input_stem}_silver.jsonl"
        
        evaluate_dataset(nlp, jobs_parquet_path, jobs_silver_path)
    except KeyError:
        logging.warning("Config for 'datasets.jobs' not found. Skipping jobs processing.")
    except Exception as e:
        logging.error(f"Error processing jobs dataset: {e}")

    try:
        resumes_cfg = config["datasets"]["resumes"]
        resumes_input_stem = Path(resumes_cfg["input_filename"]).stem
        resumes_parquet_path = processed_dir / f"{resumes_input_stem}_processed.parquet"
        resumes_silver_path = silver_dir / f"{resumes_input_stem}_silver.jsonl"

        evaluate_dataset(nlp, resumes_parquet_path, resumes_silver_path)
    except KeyError:
        logging.warning("Config for 'datasets.resumes' not found. Skipping resumes processing.")
    except Exception as e:
        logging.error(f"Error processing resumes dataset: {e}")

    logging.info("--- NER Silver Annotation Pipeline Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NER model over processed data to create 'silver' annotations.")
    parser.add_argument("--config", type=str, default="configs/config.yml", help="Path to the config.yml file")
    args = parser.parse_args()
    
    main(config_path=args.config)
