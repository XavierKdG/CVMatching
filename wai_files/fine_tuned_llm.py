import pandas as pd
from sentence_transformers import SentenceTransformer, util, InputExample
import torch
import logging
import os
import random

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
# Generate best matches + semi-hard negatives
# -----------------------------#
results = []
examples = []  # for additional fine-tuning

for i in range(len(jobs)):
    job_text = jobs.loc[i, "Job Description"]

    # Positive pair (best match)
    best_idx = int(similarity_matrix[i].argmax().item())
    best_score = float(similarity_matrix[i][best_idx].item())
    best_resume = resumes.loc[best_idx, "Resume_str"]

    results.append({
        "Job_Index": i,
        "Job ID": jobs.loc[i, "Job ID"] if "Job ID" in jobs.columns else None,
        "Job Category": jobs.loc[i, "Job Category"] if "Job Category" in jobs.columns else None,
        "Best_Resume_Index": best_idx,
        "Resume_Category": resumes.loc[best_idx, "Category"] if "Category" in resumes.columns else None,
        "Similarity": best_score,
        "Model": "MiniLM_finetuned_LoRA"
    })

    # Add positive example
    examples.append(InputExample(texts=[str(job_text), str(best_resume)], label=1.0))

    neg_resume = None
    # Hard negative: moderately similar but incorrect
    sorted_idx = torch.argsort(similarity_matrix[i], descending=True)
    found_neg = False
    for neg_idx in sorted_idx[1:10]:  # check top 10 non-best matches
        neg_score = float(similarity_matrix[i][neg_idx])
        if 0.1 <= neg_score <= 0.5:  # moderately similar
            neg_resume = resumes.loc[int(neg_idx), "Resume_str"]
            examples.append(InputExample(texts=[str(job_text), str(neg_resume)], label=0.0))
            found_neg = True
            break

    
    # Fallback: random negative if no suitable one found
    if not found_neg:
        rand_idx = random.randint(0, len(resumes) - 1)
        rand_resume = resumes.loc[rand_idx, "Resume_str"]
        examples.append(InputExample(texts=[str(job_text), str(neg_resume)], label=0.0))


# -----------------------------#
# Save matching results
# -----------------------------#
results_df = pd.DataFrame(results)
output_csv = "./data/processed/minilm_finetuned_matches.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
results_df.to_csv(output_csv, index=False)

avg_sim = results_df["Similarity"].mean()
logging.info(f"✅ Saved fine-tuned MiniLM match results to: {output_csv}")
logging.info(f"✅ Average best-match similarity: {avg_sim:.4f}")

# -----------------------------#
# Save fine-tuned model
# -----------------------------#
save_model_path = "./models/minilm_finetuned_lora_final"
os.makedirs(save_model_path, exist_ok=True)
model.save(save_model_path)
logging.info(f"✅ Fine-tuned MiniLM model saved at: {save_model_path}")
