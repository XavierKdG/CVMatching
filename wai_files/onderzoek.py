import pandas as pd

# Load CSV
df = pd.read_csv("./data/processed/job_descriptions_cleaned2.csv")

print(df["Job Description"])


