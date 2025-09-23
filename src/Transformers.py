import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
import pandas as pd

df_jobs = pd.read_csv("./data/processed/job_descriptions2_cleaned.csv")
df_resumes = pd.read_csv("./data/raw/Resume.csv")

job_texts = df_jobs["Job Description"].astype(str).tolist()
resume_texts = df_resumes["Resume_str"].astype(str).tolist() 

model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode embeddings
job_embeddings = model.encode(job_texts, convert_to_tensor=True)
resume_embeddings = model.encode(resume_texts, convert_to_tensor=True)

similarity_matrix = util.cos_sim(resume_embeddings, job_embeddings)

# Convert to DataFrame
similarity_df = pd.DataFrame(
    similarity_matrix.cpu().numpy(),
    index=[f"Resume_{i}" for i in range(len(resume_texts))],
    columns=[f"Job_{j}" for j in range(len(job_texts))]
)

# Save results
similarity_df.to_csv("./data/processed/resume_job_similarity.csv")

print("✅ Similarity matrix saved at './data/processed/resume_job_similarity.csv'")