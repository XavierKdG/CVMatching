import pandas as pd

# Load the similarity matrix you saved
similarity_df = pd.read_csv("./data/processed/resume_job_similarity.csv", index_col=0)

# Show first few rows
print(similarity_df.head())

# If you want to see which job each resume matches best:
best_matches = similarity_df.idxmax(axis=1)   # column with highest similarity
best_scores = similarity_df.max(axis=1)       # highest similarity value

results = pd.DataFrame({
    "Best_Match_Job": best_matches,
    "Best_Score": best_scores
})

print(results.head())
