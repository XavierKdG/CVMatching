# scripts/minilm_model.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from logger import setup_logger

# 1️⃣ Setup logger
logger = setup_logger(name="GlobalLogger", log_file="./logs/all_processes.log")


def generate_minilm_embeddings(job_csv="./data/processed/job_descriptions2cleaned.csv",
                               resume_csv="./data/raw/Resume.csv",
                               job_output="./data/processed/job_minilm_embeddings.csv",
                               resume_output="./data/processed/resume_minilm_embeddings.csv",
                               limit=1500):

    logger.info("🚀 Loading data...")
    jobs = pd.read_csv(job_csv).head(limit)
    resumes = pd.read_csv(resume_csv).head(limit)

    job_texts = jobs["Job Description"].astype(str).tolist()
    resume_texts = resumes["Resume_str"].astype(str).tolist()

    logger.info("🔧 Loading MiniLM model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info("📝 Generating MiniLM embeddings...")
    job_embeddings = model.encode(job_texts, convert_to_numpy=True, normalize_embeddings=True)
    resume_embeddings = model.encode(resume_texts, convert_to_numpy=True, normalize_embeddings=True)

    pd.DataFrame(job_embeddings).to_csv(job_output, index=False)
    pd.DataFrame(resume_embeddings).to_csv(resume_output, index=False)
    logger.info(f"✅ Saved embeddings at:\n  {job_output}\n  {resume_output}")

if __name__ == "__main__":
    generate_minilm_embeddings()
