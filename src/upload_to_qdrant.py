import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
import os
import argparse
import torch
from sentence_transformers import SentenceTransformer
from src.utils import setup_logging, load_config

class QdrantUploader:
    """Uploads embeddings to Qdrant from a DataFrame."""
    def __init__(self, qdrant_url='http://localhost:6333', timeout=30):
        self.client = QdrantClient(url=qdrant_url, timeout=timeout)
        logging.info(f"Qdrant connection initialized with timeout {timeout}s.")

    def upload_dataframe(self, df: pd.DataFrame, collection_name: str, payload_columns: list, batch_size: int = 256):
        """Uploads a DataFrame with embeddings directly to Qdrant in smaller batches."""
        logging.info(f"Starting upload for collection '{collection_name}' from DataFrame (batch size={batch_size})")

        extra_cols = [col for col in ['tokens', 'entities'] if col in df.columns]
        all_payload_cols = list(set(payload_columns + extra_cols))
        logging.info(f"Using payload columns: {all_payload_cols}")

        def force_list(x):
            if hasattr(x, 'tolist'): return x.tolist()
            return [float(v) for v in x]
        vectors = df['embeddings'].apply(force_list).tolist()
        
        df_reset = df.reset_index(drop=True)
        ids = df_reset.index.tolist()
        df_payload = df_reset[all_payload_cols]

        def convert_numpy_to_list(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, dict): return {k: convert_numpy_to_list(v) for k, v in obj.items()}
            if isinstance(obj, list): return [convert_numpy_to_list(i) for i in obj]
            return obj
        payload = [convert_numpy_to_list(p) for p in df_payload.to_dict(orient='records')]
        
        if not vectors:
            logging.warning(f"No vectors to upload for collection '{collection_name}'. Skipping.")
            return
            
        embedding_dim = len(vectors[0])

        try:
            if self.client.collection_exists(collection_name):
                logging.warning(f"Collection '{collection_name}' already exists and will be deleted.")
                self.client.delete_collection(collection_name)

            logging.info(f"Creating collection '{collection_name}' with vector size {embedding_dim}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE)
            )
        except Exception as e:
            logging.error(f"Error managing Qdrant collection: {e}")
            raise

        logging.info(f"Uploading {len(vectors)} points in batches of {batch_size}...")
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_vectors = vectors[i:i+batch_size]
            batch_payloads = payload[i:i+batch_size]
            
            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=models.Batch(
                        ids=batch_ids,
                        vectors=batch_vectors,
                        payloads=batch_payloads
                    ),
                    wait=True
                )
                logging.info(f"Uploaded batch {i//batch_size + 1}/{(len(ids) - 1) // batch_size + 1}")
            except Exception as e:
                logging.error(f"Error uploading batch {i//batch_size + 1}: {e}")

        logging.info(f"Upload completed for '{collection_name}' ({len(vectors)} points)")

def load_similarity_model(config):
    """Loads the SentenceTransformer model specified in the config."""
    model_name = config["models"]["similarity_model"]
    logging.info(f"Loading similarity model: {model_name}")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_name, device=device)
        logging.info("Similarity model loaded.")
        return model
    except Exception as e:
        logging.error(f"Failed to load SentenceTransformer model '{model_name}'. {e}")
        logging.error("Make sure 'sentence-transformers' is installed: pip install sentence-transformers")
        return None

def _generate_embeddings(df, model):
    """Generates embeddings from 'cleaned_text' and adds them to the DataFrame."""
    if "cleaned_text" not in df.columns:
        logging.error("DataFrame is missing 'cleaned_text' column. Skipping embedding generation.")
        return None
        
    logging.info(f"Generating embeddings for {len(df)} documents...")
    texts = df["cleaned_text"].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    
    df["embeddings"] = list(embeddings) 
    logging.info("Embeddings generated.")
    return df

def process_dataset_for_upload(dataset_name, dataset_cfg, uploader, model, config):
    """Loads, processes, and uploads a single dataset to Qdrant."""
    processed_folder = config['paths']['processed_folder']
    batch_size = config['qdrant'].get('batch_size', 256)

    input_file_stem = os.path.splitext(dataset_cfg['input_filename'])[0]
    parquet_file = f"{input_file_stem}_processed.parquet" 
    parquet_path = os.path.join(processed_folder, parquet_file)

    collection_name = dataset_cfg.get('qdrant_collection_name')
    payload_columns = dataset_cfg.get('qdrant_payload_columns', [])

    if not os.path.exists(parquet_path):
        logging.error(f"File not found, cannot upload: {parquet_path}")
        return
    if not collection_name:
        logging.warning(f"Skipping '{dataset_name}' because 'qdrant_collection_name' is not defined.")
        return
        
    logging.info(f"Processing {parquet_path} for collection '{collection_name}'")
    df = pd.read_parquet(parquet_path)

    df_with_embeddings = _generate_embeddings(df, model)

    if df_with_embeddings is not None:
        uploader.upload_dataframe(df_with_embeddings, collection_name, payload_columns, batch_size)
    else:
        logging.error(f"Skipping upload for '{collection_name}' due to embedding generation failure.")

def run_upload_pipeline(config):
    """
    Initializes the uploader, loads data, generates embeddings,
    and processes all datasets defined in the config.
    """
    qdrant_config = config['qdrant']
    uploader = QdrantUploader(qdrant_url=qdrant_config['url'], timeout=qdrant_config.get('timeout', 60))
    model = load_similarity_model(config)
    
    if model is None:
        logging.error("Aborting upload pipeline due to model loading failure.")
        return

    for dataset_name, dataset_cfg in config['datasets'].items():
        process_dataset_for_upload(
            dataset_name=dataset_name,
            dataset_cfg=dataset_cfg,
            uploader=uploader,
            model=model,
            config=config
        )

def main(config_path='configs/config.yml'):    
    config = load_config(config_path)
    run_upload_pipeline(config)
    logging.info("--- Qdrant upload pipeline complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yml", help="Path to the config.yml file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    setup_logging(config["logging"]["file_name"]) 
    
    main(config_path=args.config)