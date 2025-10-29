import pandas as pd
from sentence_transformers import SentenceTransformer, util
import logging
import os

# -----------------------------#
# Logging setup
# -----------------------------#
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------#
# Load fine-tuned MiniLM model
# -----------------------------#
model_path = "./models/minilm_finetuned_lora"
logging.info(f"Loading fine-tuned model from {model_path} ...")
model = SentenceTransformer(model_path)

# -----------------------------#
# Load datasets
# -----------------------------#
jobs = pd.read_csv("./data/processed/job_descriptions2cleaned.csv")
resumes = pd.read_csv("./data/raw/Resume.csv")

jobs["Job Description"] = jobs["Job Description"].astype(str)
resumes["Resume_str"] = resumes["Resume_str"].astype(str)

# -----------------------------#
# Encode job + resume texts
# -----------------------------#
logging.info("Encoding job descriptions and resumes (batched)...")
job_embeddings = model.encode(jobs["Job Description"].tolist(), convert_to_tensor=True, show_progress_bar=True)
resume_embeddings = model.encode(resumes["Resume_str"].tolist(), convert_to_tensor=True, show_progress_bar=True)

# -----------------------------#
# Compute cosine similarity
# -----------------------------#
logging.info("Computing cosine similarities...")
similarity_matrix = util.cos_sim(job_embeddings, resume_embeddings)

# -----------------------------#
# Find best matching resume per job
# -----------------------------#
results = []
for i in range(len(jobs)):
    best_idx = int(similarity_matrix[i].argmax().item())
    best_score = float(similarity_matrix[i][best_idx].item())

    results.append({
        "Job_Index": i,
        "Job ID": jobs.loc[i, "Job ID"] if "Job ID" in jobs.columns else None,
        "Job Category": jobs.loc[i, "Job Category"] if "Job Category" in jobs.columns else None,
        "Best_Resume_Index": best_idx,
        "Resume_Category": resumes.loc[best_idx, "Category"] if "Category" in resumes.columns else None,
        "Similarity": best_score,
        "Model": "MiniLM_finetuned_LoRA"
    })

# -----------------------------#
# Save results
# -----------------------------#
results_df = pd.DataFrame(results)
output_path = "./data/processed/minilm_finetuned_matches.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
results_df.to_csv(output_path, index=False)

avg_sim = results_df["Similarity"].mean()
logging.info(f"✅ Saved fine-tuned MiniLM match results to: {output_path}")
logging.info(f"✅ Average best-match similarity: {avg_sim:.4f}")
