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
                               model_path="./models/minilm_model",
                               limit=1500):

    logger.info("🚀 Loading data...")
    jobs = pd.read_csv(job_csv).head(limit)
    resumes = pd.read_csv(resume_csv).head(limit)

    job_texts = jobs["Job Description"].astype(str).tolist()
    resume_texts = resumes["Resume_str"].astype(str).tolist()

    # 2️⃣ Load MiniLM
    logger.info("🔧 Loading MiniLM model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3️⃣ Save model locally for reuse
    logger.info(f"💾 Saving MiniLM model to {model_path} ...")
    model.save(model_path)
    logger.info(f"✅ MiniLM model saved at: {model_path}")

    # 4️⃣ Generate embeddings
    logger.info("📝 Generating MiniLM embeddings...")
    job_embeddings = model.encode(job_texts, convert_to_numpy=True, normalize_embeddings=True)
    resume_embeddings = model.encode(resume_texts, convert_to_numpy=True, normalize_embeddings=True)

    # 5️⃣ Save embeddings
    pd.DataFrame(job_embeddings).to_csv(job_output, index=False)
    pd.DataFrame(resume_embeddings).to_csv(resume_output, index=False)
    logger.info(f"✅ Saved embeddings at:\n  {job_output}\n  {resume_output}")

if __name__ == "__main__":
    generate_minilm_embeddings()
