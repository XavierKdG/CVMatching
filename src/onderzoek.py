import pandas as pd

# Load CSV
results_df = pd.read_csv("./data/processed/job_best_resume_match.csv")

# Find the best match for each Resume_Index
best_per_resume = results_df.loc[results_df.groupby("Resume_Index")["Similarity"].idxmax()]

# Reset index for neatness
best_per_resume = best_per_resume.reset_index(drop=True)

# Save results if needed
best_per_resume.to_csv("./data/processed/best_job_per_resume.csv", index=False)

# Print results
for _, row in best_per_resume.iterrows():
    print(f"🧑 Resume {row['Resume_Index']} ({row['Resume_Category']})")
    print(f"   👉 Best Job: {row['Job_Category']}")
    print(f"   🔥 Similarity: {row['Similarity']:.4f}\n")
