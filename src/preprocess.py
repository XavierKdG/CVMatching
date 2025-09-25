import pandas as pd
import os 
import re
import html
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)

def load_raw_csv(file_path):
    return pd.read_csv(file_path)

def clean_text(text):
    if pd.isna(text):
        return ""
    
    # Fix encoding issues (like â€™ to ’)
    text = text.encode("latin1", "ignore").decode("utf-8", "ignore")
    text = html.unescape(text)

    # 2. Remove HTML tags if any
    text = re.sub(r"<.*?>", " ", text)

    # 3. Normalize whitespace and replace newlines/tabs
    text = re.sub(r"\s+", " ", text)

    # 4. Remove bullet points, dashes, special characters but keep .,!? for readability
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)

    # 5. Remove multiple punctuation (e.g. "!!!" -> "!")
    text = re.sub(r"([.,!?])\1+", r"\1", text)

    # 6. Convert to lowercase
    text = text.lower().strip()

    # 7. Remove English stopwords
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    text = pattern.sub("", text)

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