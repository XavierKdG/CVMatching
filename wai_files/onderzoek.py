import pandas as pd

# Load CSV
df = pd.read_csv("/home/admin-groep21/CVMatching/data/raw/job_descriptions.csv")
print(df.columns)

num_jobid_duplicates = df.duplicated(subset=["Job Id"]).sum()
print(f"Number of duplicate Job Ids: {num_jobid_duplicates}")




