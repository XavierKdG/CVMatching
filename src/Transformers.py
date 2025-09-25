import pandas as pd
from sentence_transformers import SentenceTransformer
    
# Load datasets
df_jobs = pd.read_csv("./data/processed/job_descriptions_cleaned.csv").head(5000)  # limit jobs
df_resumes = pd.read_csv("./data/raw/Resume.csv")  # limit resumes

# Columns
job_texts = df_jobs["Job Description"].astype(str).tolist()
resume_texts = df_resumes["Resume_str"].astype(str).tolist()

job_categories = df_jobs["Job Title"].astype(str).tolist()
resume_categories = df_resumes["Category"].astype(str).tolist()

# Add Job ID if available (optional, fallback if missing)
job_ids = df_jobs["Job Id"].astype(str).tolist() if "Job Id" in df_jobs.columns else list(range(len(df_jobs)))

# Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embeddings with normalization
job_embeddings = model.encode(job_texts, convert_to_tensor=True, normalize_embeddings=True)
resume_embeddings = model.encode(resume_texts, convert_to_tensor=True, normalize_embeddings=True)

# Direct dot product (cosine similarity)
similarity_matrix = job_embeddings @ resume_embeddings.T

# Build results: for each job, pick best resume
results = []
for j, (job_id, job_cat, job_text) in enumerate(zip(job_ids, job_categories, job_texts)):
    scores = similarity_matrix[j].cpu().numpy()
    best_idx = scores.argmax()  # best resume index
    results.append({
        "Job_ID": job_id,
        "Job_Category": job_cat,
        "Resume_Category": resume_categories[best_idx],
        "Similarity": scores[best_idx]
    })

results_df = pd.DataFrame(results)
results_df.to_csv("./data/processed/full_job_best_resume_match2.csv", index=False)
print("✅ Saved at ./data/processed/full_job_best_resume_match2.csv")
