# scripts/doc2vec_model.py
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import normalize
from logger import setup_logger

# 1️⃣ Setup logger
logger = setup_logger(name="GlobalLogger", log_file="./logs/all_processes.log")

def train_doc2vec(job_csv="./data/processed/job_descriptions2cleaned.csv",
                  resume_csv="./data/raw/Resume.csv",
                  job_output="./data/processed/job_doc2vec_embeddings.csv",
                  resume_output="./data/processed/resume_doc2vec_embeddings.csv",
                  model_path="./models/doc2vec_model.model",
                  limit=1500):

    logger.info("🚀 Loading data...")
    jobs = pd.read_csv(job_csv).head(limit)
    resumes = pd.read_csv(resume_csv).head(limit)

    job_docs = jobs["Job Description"].astype(str).tolist()
    resume_docs = resumes["Resume_str"].astype(str).tolist()

    all_docs = job_docs + resume_docs
    tagged_data = [TaggedDocument(words=doc.split(), tags=[f'doc_{i}'])
                   for i, doc in enumerate(all_docs)]

    logger.info("🔧 Training Doc2Vec model...")
    model = Doc2Vec(tagged_data, vector_size=100, window=5, min_count=2, workers=4, epochs=50)
    model.save(model_path)
    logger.info(f"✅ Model saved at {model_path}")

    # Extract embeddings
    logger.info("📦 Extracting embeddings...")
    job_emb = np.array([model.dv[f'doc_{i}'] for i in range(len(job_docs))])
    resume_emb = np.array([model.dv[f'doc_{i}'] for i in range(len(job_docs), len(all_docs))])

    # ✅ Normalize embeddings (important for cosine similarity)
    job_emb = normalize(job_emb)
    resume_emb = normalize(resume_emb)

    pd.DataFrame(job_emb).to_csv(job_output, index=False)
    pd.DataFrame(resume_emb).to_csv(resume_output, index=False)
    logger.info(f"✅ Saved normalized embeddings at:\n  {job_output}\n  {resume_output}")

if __name__ == "__main__":
    train_doc2vec()
