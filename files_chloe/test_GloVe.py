# Installatie (eenmalig)
# pip install pandas numpy scikit-learn tqdm

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# --- 1️⃣ Data inladen ---
df = pd.read_csv("/home/admin-groep11/CVMatching-1/data/processed/labeled_jobdescriptions2_cleaned.csv")
# Kolommen: 'resume_text', 'job_text', 'label'

X_train, X_test, y_train, y_test = train_test_split(
    df[['resume_text', 'job_text']], df['label'], test_size=0.2, random_state=42
)

glove_path = "/home/admin-groep11/CVMatching-1/glove.6B.100d.txt"

print(">>> GloVe laden...")
embeddings_index = {}
with open(glove_path, 'r', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = vector
print(f"GloVe woorden geladen: {len(embeddings_index):,}")

embedding_dim = 100

def text_to_vector(text):
    words = str(text).lower().split()
    word_vecs = [embeddings_index[w] for w in words if w in embeddings_index]
    if len(word_vecs) == 0:
        return np.zeros(embedding_dim)
    return np.mean(word_vecs, axis=0)

print(">>> Embedden van teksten...")
resume_emb_train = np.vstack([text_to_vector(t) for t in tqdm(X_train['resume_text'])])
job_emb_train = np.vstack([text_to_vector(t) for t in tqdm(X_train['job_text'])])
resume_emb_test = np.vstack([text_to_vector(t) for t in tqdm(X_test['resume_text'])])
job_emb_test = np.vstack([text_to_vector(t) for t in tqdm(X_test['job_text'])])

X_train_emb = np.hstack([resume_emb_train, job_emb_train])
X_test_emb = np.hstack([resume_emb_test, job_emb_test])

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),  # 3 lagen
    activation='relu',
    solver='adam',
    batch_size=32,
    max_iter=100,
    random_state=42
)

print(">>> Train MLPClassifier...")
mlp.fit(X_train_emb, y_train)

y_pred = mlp.predict(X_test_emb)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n Evaluatie resultaten:")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
