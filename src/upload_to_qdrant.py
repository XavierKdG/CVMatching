import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
import os
# Make sure this import path is correct for your project structure
from src.utils import setup_logging, load_config

class QdrantUploader:
    """Uploads embeddings to Qdrant from a DataFrame or a Parquet file."""
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
        
        ids = df.index.tolist()
        df_payload = df[all_payload_cols]

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

        if self.client.collection_exists(collection_name):
            logging.warning(f"Collection '{collection_name}' already exists and will be deleted.")
            self.client.delete_collection(collection_name)

        logging.info(f"Creating collection '{collection_name}' with vector size {embedding_dim}")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE)
        )

        logging.info(f"Uploading {len(vectors)} points in batches of {batch_size}...")
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_vectors = vectors[i:i+batch_size]
            batch_payloads = payload[i:i+batch_size]
            
            self.client.upsert(
                collection_name=collection_name,
                points=models.Batch(
                    ids=batch_ids,
                    vectors=batch_vectors,
                    payloads=batch_payloads
                ),
                wait=True
            )
            logging.info(f"Uploaded batch {i//batch_size + 1}...")

        logging.info(f"Upload completed for '{collection_name}' ({len(vectors)} points)")

    def upload_collection(self, parquet_path, collection_name, payload_columns, batch_size=256):
        """(Legacy) Uploads a Parquet file by reading it into a DataFrame first."""
        logging.info(f"Reading data from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        self.upload_dataframe(df, collection_name, payload_columns, batch_size)

def run_upload_pipeline(config):
    """
    Initializes the uploader and processes all datasets defined in the config.
    """
    qdrant_config = config['qdrant']
    uploader = QdrantUploader(qdrant_url=qdrant_config['url'], timeout=qdrant_config.get('timeout', 60))
    processed_folder = config['paths']['processed_folder']
    batch_size = qdrant_config.get('batch_size', 256)

    for dataset_name, dataset_cfg in config['datasets'].items():
        input_file_stem = os.path.splitext(dataset_cfg['input_filename'])[0]
        parquet_file = f"{input_file_stem}_embeddings.parquet"
        parquet_path = os.path.join(processed_folder, parquet_file)

        collection_name = dataset_cfg.get('qdrant_collection_name')
        payload_columns = dataset_cfg.get('qdrant_payload_columns', [])

        if os.path.exists(parquet_path) and collection_name:
            uploader.upload_collection(parquet_path, collection_name, payload_columns, batch_size)
        elif not collection_name:
             logging.warning(f"Skipping '{dataset_name}' because 'qdrant_collection_name' is not defined.")
        else:
            logging.error(f"File not found, cannot upload: {parquet_path}")

def main(config_path='config.yml'):
    config = load_config(config_path)
    run_upload_pipeline(config)
    logging.info("--- Successfully uploaded to Qdrant ---")

if __name__ == "__main__":
    setup_logging() 
    main()