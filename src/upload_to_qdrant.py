import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
import os
from src.utils import setup_logging, load_config

class QdrantUploader:
    """Uploads Parquet embeddings to Qdrant."""
    def __init__(self, qdrant_url='http://localhost:6333', timeout=20):
        self.client = QdrantClient(url=qdrant_url, timeout=timeout)
        logging.info(f"Qdrant connection initialized with timeout {timeout}s.")

    def upload_collection(self, parquet_path, collection_name, payload_columns, batch_size=128):
        logging.info(f"Starting upload for collection '{collection_name}' (batch size={batch_size})")
        df = pd.read_parquet(parquet_path)

        def force_list(x):
            if hasattr(x, 'tolist'):
                x = x.tolist()
            return [float(v) for v in x]

        vectors = df['embeddings'].apply(force_list).tolist()

        ids = df.index.tolist()
        df_payload = df[payload_columns]
 
        def convert_numpy_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_numpy_to_list(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy_to_list(i) for i in obj]
            return obj

        payload = df_payload.to_dict(orient='records')
        payload = [convert_numpy_to_list(p) for p in payload]
        
        embedding_dim = len(vectors[0])

        if self.client.collection_exists(collection_name):
            logging.warning(f"Collection '{collection_name}' exists and will be deleted.")
            self.client.delete_collection(collection_name)

        logging.info(f"Creating collection '{collection_name}'")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE)
        )

        logging.info(f"Uploading {len(vectors)} points in batches...")
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_vectors = vectors[i:i+batch_size]
            batch_payloads = payload[i:i+batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=models.Batch(ids=batch_ids, vectors=batch_vectors, payloads=batch_payloads),
                wait=True
            )

        logging.info(f"Upload completed for '{collection_name}' ({len(vectors)} points)")

def run_upload_pipeline(config):
    qdrant_config = config['qdrant']
    uploader = QdrantUploader(qdrant_url=qdrant_config['url'], timeout=qdrant_config.get('timeout', 20))
    processed_folder = config['data']['processed_folder']
    batch_size = qdrant_config.get('batch_size', 128)

    for key, col_cfg in qdrant_config['collections'].items():
        parquet_path = os.path.join(processed_folder, col_cfg['path'])
        if os.path.exists(parquet_path):
            uploader.upload_collection(parquet_path, col_cfg['name'], col_cfg['payload_columns'], batch_size)
        else:
            logging.error(f"File not found: {parquet_path}")

def main(config_path=None):
    setup_logging()
    config = load_config(config_path)
    run_upload_pipeline(config)
    logging.info("--- Successfully uploaded to Qdrant ---")

if __name__ == "__main__":
    main()
