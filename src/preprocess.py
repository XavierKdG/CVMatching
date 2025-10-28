import pandas as pd
import os 
import re
import spacy
import argparse
import logging
import json
from spacy.tokens import DocBin
from sklearn.model_selection import train_test_split
from .utils import setup_logging, load_config #help functions

class Preprocessor:
    def __init__(self, config):
        """Initialize the Preprocessor class with config and NLP models."""
        logging.info("Initializing Preprocessor...")
        self.config = config
        self.raw_dir = config["paths"]["raw_folder"]
        self.processed_dir = config["paths"]["processed_folder"]

        base_model = self.config["models"]["base_model"] 
        disabled_pipes = ["parser", "ner"] #not needed for this task
        logging.info(f"Loading spaCy model for lemmatization: {base_model}")
        self.similarity_nlp = spacy.load(base_model, disable=disabled_pipes) #load nlp model

        self.ner_nlp = spacy.blank("en") #load blank model
        logging.info("Blank 'en' model for NER processing initialized.")

    def run_similarity_preprocessing(self):
        """Runs the full CSV-to-Parquet preprocessing pipeline for all datasets defined in the config."""
        logging.info("--- Starting Parquet Preprocessing (for Qdrant) ---")
        chunksize = self.config["preprocessing"].get("chunksize", 50000)
        
        self._process_file(self.config["datasets"]["jobs"], chunksize)
        self._process_file(self.config["datasets"]["resumes"], chunksize)
        logging.info("--- Parquet Preprocessing Complete ---")

    def _process_file(self, dataset_config, chunksize):
        """Internal: Processes a single CSV file in chunks."""
        file_name = dataset_config["input_filename"]
        input_path = os.path.join(self.raw_dir, file_name)
        output_filename = os.path.splitext(file_name)[0] + "_processed.parquet"
        output_path = os.path.join(self.processed_dir, output_filename)

        logging.info(f"--- Processing file: {file_name} with chunksize={chunksize} ---")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        processed_chunks = []
        try:
            for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False, encoding='utf-8'):
                processed_df = self._process_dataframe(
                    df=chunk,
                    columns_to_combine=dataset_config["columns_to_process"],
                    duplicate_subset=dataset_config.get("duplicate_subset")
                )
                processed_chunks.append(processed_df)
        except FileNotFoundError:
            logging.error(f"Input file not found at: {input_path}")
            raise
        except Exception as e:
            logging.error(f"Error processing chunks for {file_name}: {e}")
            raise

        if processed_chunks:
            pd.concat(processed_chunks).to_parquet(output_path, index=False)
            logging.info(f"File saved to {output_path}")
        else:
            logging.warning(f"No data found or processed for {file_name}.")

    def _process_dataframe(self, df, columns_to_combine, duplicate_subset=None):
        """Internal: Processes a single DataFrame chunk."""
        logging.info(f"Processing dataframe with {len(df)} rows.")
        df_copy = df.copy() 
        initial_rows = len(df_copy)

        if duplicate_subset and all(col in df_copy.columns for col in duplicate_subset):
            df_copy.drop_duplicates(subset=duplicate_subset, inplace=True)
        else: 
            df_copy.drop_duplicates(inplace=True)
        logging.info(f"{initial_rows - len(df_copy)} duplicate rows removed")

        df_copy["combined_text"] = df_copy[columns_to_combine].astype(str).agg(" ".join, axis=1)
        df_copy["cleaned_text"] = df_copy["combined_text"].apply(self._clean_text)
        
        logging.info("Tokenizing and processing texts in a batch...")
        texts_to_process = df_copy["cleaned_text"].tolist()
        
        docs = self.similarity_nlp.pipe(texts_to_process, batch_size=100)

        all_tokens = []
        for doc in docs:
            processed = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
            all_tokens.append(processed)

        df_copy["tokens"] = all_tokens
        logging.info("Batch tokenizing complete.")

        return df_copy.drop(columns=['combined_text'])

    def _clean_text(self, text):
        """Internal: Cleans a single string of text."""
        text = re.sub(r'\s+', ' ', str(text))
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        return text.strip()

    def run_ner_preprocessing(self):
        """Runs the full JSONL to spacy preprocessing pipeline. This is skipped if pipeline.mode is not 'custom'."""
        if self.config["pipeline"]["mode"] != "custom":
            logging.info("--- Skipping NER Corpus Creation (Base Mode) ---")
            return

        logging.info("--- Starting NER Corpus Creation (Custom Mode) ---")
        
        ner_data_path = os.path.join(self.raw_dir, self.config["preprocessing"]["train_data_file"])
        train_corpus_path = os.path.join(self.processed_dir, "train.spacy")
        dev_corpus_path = os.path.join(self.processed_dir, "dev.spacy")
        
        ner_data = self._load_ner_data(ner_data_path)
        if ner_data:
            self._create_spacy_corpus(
                data=ner_data,
                train_path=train_corpus_path,
                dev_path=dev_corpus_path,
                train_ratio=self.config["preprocessing"]["trainsize"],
                seed=self.config["training"]["seed"]
            )
        else:
            logging.error("Failed to load NER data. Skipping .spacy corpus creation.")
        logging.info("--- NER Corpus Creation Complete ---")

    def _load_ner_data(self, jsonl_path):
        """Internal: Loads NER data from a .jsonl file."""
        label_key = self.config["preprocessing"].get("ner_label_key", "entities")
        
        logging.info(f"Loading NER data from {jsonl_path} using label key: '{label_key}'")
        data = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)

                    entities = item.get(label_key) or []
                    
                    data.append((item['text'], entities))
            logging.info(f"Loaded {len(data)} NER records.")
            return data
        except FileNotFoundError:
            logging.error(f"NER data file not found: {jsonl_path}")
            return []
        except Exception as e:
            logging.error(f"Error loading {jsonl_path}: {e}")
            return []

    def _create_spacy_corpus(self, data, train_path, dev_path, train_ratio, seed):
        """Internal: Splits data and creates .spacy files."""
        logging.info(f"Splitting data with train_ratio={train_ratio} (seed={seed})")
        train_data, dev_data = train_test_split(data, train_size=train_ratio, random_state=seed)
        logging.info(f"Training samples: {len(train_data)}, Development samples: {len(dev_data)}")

        self._create_docbin(train_data, train_path)
        self._create_docbin(dev_data, dev_path)

    def _create_docbin(self, data, file_path):
        """Internal: Creates a single .spacy DocBin file."""
        logging.info(f"Creating DocBin at {file_path} with {len(data)} samples...")
        db = DocBin()
        
        skipped_count = 0
        for text, annotations in data:
            doc = self.ner_nlp(text)
            ents = []
            for start, end, label in annotations:
                span = doc.char_span(start, end, label=label)
                if span is None:
                    logging.warning(f"Skipping invalid span: '{text[start:end]}' in '{text[:50]}...'")
                    skipped_count += 1
                else:
                    ents.append(span)
            doc.ents = ents
            db.add(doc)
            
        db.to_disk(file_path)
        if skipped_count > 0:
            logging.warning(f"Total skipped spans for {file_path}: {skipped_count}")
        logging.info(f"Successfully created {file_path}")

def main(config_path=None):
    """Main function to initialize and run the preprocessor."""
    if config_path is None:
        config_path = "configs/config.yml" 
        
    config = load_config(config_path) 
    setup_logging(config["logging"]["file_name"])

    preprocessor = Preprocessor(config) #Initialize the general preprocessor
    preprocessor.run_similarity_preprocessing() #CSV to Parquet
    preprocessor.run_ner_preprocessing() #JSONL to .spacy
    
    logging.info("--- All Preprocessing Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yml", help="Path to the config.yml file")
    args = parser.parse_args()
    
    main(config_path=args.config)