import pandas as pd
import os 
import re
from sklearn.model_selection import train_test_split

os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)

def load_raw_csv(file_path):
    return pd.read_csv(file_path)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text = text.lower().strip()
    return text

def main():
    df1 = pd.read_csv("./data/raw/job_descriptions.csv")
    df2 = pd.read_csv("./data/raw/job_descriptions2.csv")

    df1['Job Description'] = df1['Job Description'].apply(clean_text)
    df2['Job Description'] = df2['Job Description'].apply(clean_text)
    
    df1.to_csv('./data/processed/processed_df1.csv', index=False)
    df2.to_csv('./data/processed/processed_df2.csv', index=False)

if __name__ == "__main__":
    main()