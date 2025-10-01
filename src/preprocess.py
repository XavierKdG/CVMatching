import pandas as pd
import os 
import re
import html
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')      
# nltk.download('wordnet')    
# nltk.download('omw-1.4') 
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)

def load_raw_csv(file_path):
    return pd.read_csv(file_path)

def clean_text(text):
    if pd.isna(text):
        return []
    
    # 1. Fix encoding issues (like â€™ to ’)
    text = text.encode("latin1", "ignore").decode("utf-8", "ignore")
    text = html.unescape(text)

    # 2. Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # 3. Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # 4. Remove special characters (keep only alphanumeric + space)
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)

    # 5. Lowercase
    text = text.lower().strip()

    # --- Tokenization ---
    tokens = word_tokenize(text)

    # 6. Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in stop_words]

    # 7. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return tokens   # return list instead of string

def main():
    # df1 = pd.read_csv("./data/raw/job_descriptions.csv")
    df2 = pd.read_csv("./data/raw/job_descriptions2.csv")

    # df1['Job Description'] = df1['Job Description'].apply(clean_text)
    df2['Job Description'] = df2['Job Description'].apply(clean_text)
    
    # df1.to_csv('./data/processed/processed_df1.csv', index=False)
    df2.to_csv('./data/processed/processed_df2.csv', index=False)

if __name__ == "__main__":
    main()
