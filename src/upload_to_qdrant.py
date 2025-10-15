import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
import ast
import logging
import os
from src.utils import setup_logging, load_config

class QdrantUploader:
    """Class to upload CSV files to Qdrant database"""
    def __init__(self, qdrant_url='http://localhost:6333', timeout=20):
        self.client = QdrantClient(url=qdrant_url, timeout=timeout)
        logging.info(f"Qdrant connection initialized with a timeout of {timeout} seconds.")
    
    def upload_collection(self, csv_path, collection_name, payload_columns, batch_size=128):
        """Leest een CSV, (her)creëert een collectie en uploadt de data in batches."""
        logging.info(f"Starting upload for collection: '{collection_name}' with batch size {batch_size}")
        
        df = pd.read_csv(csv_path)
        vectors = df['embeddings'].apply(ast.literal_eval).tolist()
        ids = df.index.tolist()
        embedding_dim = len(vectors[0])
        df_payload = df.drop(columns=['embeddings'])
        payload = df_payload[payload_columns].to_dict(orient='records')

        if self.client.collection_exists(collection_name):
            logging.warning(f"Collection '{collection_name}' already exists and will be deleted.")
            self.client.delete_collection(collection_name)
        logging.info(f"Creating new collection: '{collection_name}'")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE)
        )

        logging.info(f"Starting to upload {len(vectors)} points in batches of {batch_size}...")

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_vectors = vectors[i:i + batch_size]
            batch_payloads = payload[i:i + batch_size]
            
            self.client.upsert(
                collection_name=collection_name,
                points=models.Batch(
                    ids=batch_ids,
                    vectors=batch_vectors,
                    payloads=batch_payloads
                ),
                wait=True 
            )

        logging.info(f"Upload of {len(vectors)} embeddings to Qdrant collection '{collection_name}' completed.")

def run_upload_pipeline(config):
    """Run the whole uploading pipeline"""
    qdrant_config = config['qdrant']
    uploader = QdrantUploader(
        qdrant_url=qdrant_config['url'],
        timeout=qdrant_config.get('timeout', 20)
    )
    processed_folder = config['data']['processed_folder']
    batch_size = qdrant_config.get('batch_size', 128)

    for key, collection_config in qdrant_config['collections'].items():
        file_name = collection_config['path']
        csv_path = os.path.join(processed_folder, file_name)
        
        if os.path.exists(csv_path):
            uploader.upload_collection(
                csv_path=csv_path,
                collection_name=collection_config['name'],
                payload_columns=collection_config['payload_columns'],
                batch_size=batch_size 
            )
        else:
            logging.error(f"File not found, upload skipped: {csv_path}")
    
def main():
    """Main function to upload to Qdrant"""
    config = load_config()
    run_upload_pipeline(config)
    logging.info("--- Succesfully uploaded to Qdrant ---")

if __name__ == "__main__":
    main()
    setup_logging()
