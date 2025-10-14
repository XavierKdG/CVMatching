import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# === 1. Laad je dataset ===
df = pd.read_csv("/home/admin-groep11/CVMatching-1/data/processed/labeled_jobdescriptions2_cleaned.csv")

# Controleer of kolom bestaat
print("Kolommen in dataset:", df.columns.tolist())

# === 2. TF-IDF vectoriseren van job_text ===
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X = vectorizer.fit_transform(df["job_text"])

print("\n✅ TF-IDF matrix aangemaakt.")
print("Vorm van matrix:", X.shape)

# === 3. Top 20 woorden (gemiddelde TF-IDF score) ===
feature_names = vectorizer.get_feature_names_out()
mean_scores = X.mean(axis=0).A1
top_indices = np.argsort(mean_scores)[::-1][:20]

print("\n🔝 Top 20 woorden volgens TF-IDF:")
for idx in top_indices:
    print(f"{feature_names[idx]} ({mean_scores[idx]:.4f})")

# === 4. Controle op lege documenten ===
empty_docs = (X.sum(axis=1) == 0).sum()
print(f"\n🧐 Aantal lege documenten: {empty_docs}")

# === 5. Cosine similarity tussen eerste 10 vacatures ===
sim_matrix = cosine_similarity(X[:10])
print("\nCosine similarity (eerste 10 documenten):")
print(np.round(sim_matrix, 2))

# === 6. Optionele visualisatie ===
X_dense = X.toarray()
pca = PCA(n_components=2, random_state=42)
reduced = pca.fit_transform(X_dense[:300])  # 300 voor snelheid

plt.figure(figsize=(8,6))
plt.scatter(reduced[:,0], reduced[:,1], alpha=0.6, color="steelblue")
plt.title("TF-IDF 2D visualisatie via PCA (job_text)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
