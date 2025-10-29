import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from logger import setup_logger

logger = setup_logger(name="GlobalLogger", log_file="./logs/all_processes.log")

def compute_matches(
    job_emb_path,
    resume_emb_path,
    job_csv="./data/processed/job_descriptions2cleaned.csv",
    resume_csv="./data/raw/Resume.csv",
    output_csv="./data/processed/matches_with_descriptions.csv",
    model_name="Doc2Vec"
):
    logger.info(f"🚀 Loading embeddings for {model_name}...")
    job_emb = pd.read_csv(job_emb_path).to_numpy()
    resume_emb = pd.read_csv(resume_emb_path).to_numpy()

    jobs = pd.read_csv(job_csv).reset_index().rename(columns={"index": "Job_Index"})
    resumes = pd.read_csv(resume_csv).reset_index().rename(columns={"index": "Resume_Index"})

    logger.info("📈 Computing cosine similarity matrix...")
    similarity_matrix = cosine_similarity(job_emb, resume_emb)

    results = []
    for i in range(len(job_emb)):
        best_idx = similarity_matrix[i].argmax()
        results.append({
            "Job_Index": i,
            "Job ID": jobs.loc[i, "Job ID"] if "Job ID" in jobs.columns else None,
            "Job Category": jobs.loc[i, "Job Category"] if "Job Category" in jobs.columns else None,
            "Job Description": jobs.loc[i, "Job Description"],
            "Best_Resume_Index": best_idx,
            "Resume_Category": resumes.loc[best_idx, "Category"] if "Category" in resumes.columns else None,
            "Similarity": float(similarity_matrix[i][best_idx]),
            "Model": model_name
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    logger.info(f"✅ Saved matches with job descriptions at: {output_csv}")
    return df_results


if __name__ == "__main__":
    # Example for Doc2Vec
    compute_matches(
        job_emb_path="./data/processed/job_doc2vec_embeddings.csv",
        resume_emb_path="./data/processed/resume_doc2vec_embeddings.csv",
        output_csv="./data/processed/doc2vec_job_resume_matches.csv",
        model_name="Doc2Vec"
    )

    # Example for MiniLM
    compute_matches(
        job_emb_path="./data/processed/job_minilm_embeddings.csv",
        resume_emb_path="./data/processed/resume_minilm_embeddings.csv",
        output_csv="./data/processed/minilm_job_resume_matches.csv",
        model_name="MiniLM"
    )
