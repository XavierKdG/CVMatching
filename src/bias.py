import pandas as pd

df = pd.read_csv("./data/raw/job_descriptions2.csv")
df[["Business Title", "Preferred Skills"]].head(10)
print(df[["Business Title", "Preferred Skills"]].head(100))