import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Sample documents (simple text snippets for demonstration)
df = pd.read_csv("./data/processed/job_descriptions_cleaned2.csv")

df_sample = df.head(5000)

documents = df_sample["Job Description"].astype(str).tolist()
# Preprocess documents for Doc2Vec
tagged_data = [TaggedDocument(words=doc.split(), tags=[f'doc_{i}']) for i, doc in enumerate(documents)]

# Train a Doc2Vec model
model = Doc2Vec(tagged_data, vector_size=10, window=2, min_count=1, workers=4, epochs=100)

# Extract vectors for each document
doc_vectors = np.array([model.dv[f'doc_{i}'] for i in range(len(documents))])

# Use t-SNE to reduce dimensionality of document vectors for visualization
tsne_model = TSNE(n_components=2, random_state=42, perplexity=3)
reduced_vectors = tsne_model.fit_transform(doc_vectors)

# Plot the 2D representation of documents
plt.figure(figsize=(8, 6))
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], color='blue')

# Annotate the points with document labels
for i, doc in enumerate(documents[:20]):
    plt.annotate(f"Doc {i+1}", (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=12)

# Title and axes labels
plt.title("Document Embeddings Visualization (Doc2Vec + t-SNE)", fontsize=16)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

# Display the plot
plt.show()