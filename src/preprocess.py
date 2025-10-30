import pandas as pd
import os
import re
import html
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# Create folders if they don’t exist
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)

# ----------------------
# TEXT CLEANING
# ----------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    
    # 1. Fix encoding issues
    text = text.encode("latin1", "ignore").decode("utf-8", "ignore")
    text = html.unescape(text)

    # 2. Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # 3. Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # 4. Remove special characters (keep only alphanumeric + space)
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)

    # 5. Lowercase
    return text.lower().strip()


# ----------------------
# LOAD BERT MODEL
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(device)


# ----------------------
# BATCH EMBEDDING FUNCTION
# ----------------------
def get_embeddings_batch(texts, batch_size=32, max_length=128):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        # Tokenize batch
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                           padding=True, max_length=max_length).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # CLS token embedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.extend(cls_embeddings)

    print(f"Processed batch {i // batch_size + 1}")
    
    return np.array(all_embeddings)


# ----------------------
# MAIN PIPELINE
# ----------------------
def main():
    # Load raw dataset
    df2 = pd.read_csv("./data/raw/job_descriptions2.csv")

    # Step 1: Clean text
    df2['Job Description'] = df2['Job Description'].apply(clean_text)

    # Step 2: BERT embeddings
    texts = df2['Job Description'].tolist()
    embeddings = get_embeddings_batch(texts, batch_size=32)

    # Step 3: Save embeddings
    df2['BERT_Embeddings'] = embeddings.tolist()  # list for CSV readability

    df2.to_pickle('./data/processed/processed_df2_with_embeddings.pkl')  # preserves NumPy arrays
    df2.to_csv('./data/processed/processed_df2.csv', index=False)

    print("✅ Processing complete! Saved to data/processed/")

if __name__ == "__main__":
    main()
