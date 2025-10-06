# src/upload_to_qdrant.py
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
import ast

CSV_FILE = 'data/processed/job_descriptions_embeddings.csv'
COLLECTION_NAME = 'job_embeddings'
QDRANT_URL = 'http://localhost:6333'

df = pd.read_csv(CSV_FILE)
df['tokens'] = df['tokens'].apply(ast.literal_eval)

vectors = df['embeddings'].apply(ast.literal_eval) if isinstance(df['embeddings'][0], str) else df['embeddings'].tolist()
ids = df.index.tolist()
payload = df[['Job Category', 'Job ID']].to_dict(orient='records')

client = QdrantClient(url=QDRANT_URL)
embedding_dim = len(vectors[0])

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=embedding_dim,
        distance=models.Distance.COSINE
    )
)

client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
        models.PointStruct(id=ids[i], vector=vectors[i], payload=payload[i])
        for i in range(len(vectors))
    ]
)

print(f"Uploaded {len(vectors)} embeddings to Qdrant collection '{COLLECTION_NAME}'")
