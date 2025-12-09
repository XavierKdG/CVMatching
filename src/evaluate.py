#notes
# om te evulueren zijn er 2 methodes om het te doen. We gaan het labelen en dan evaluaten op basis van de labels.
# semantic search is de andere methode om het te doen. zonder labels.

from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import sklearn.metrics

# 1 evaluatie scoring methode van het model.
# silhouette score zegt hoe goed de embeddings zijn geclusterd en indirect hoe goed de model data onderscheidt in clusters
# resultaat tussen -1 en 1.
def silhouette_scores(embedding_matrix: np.ndarray):
    analyse = np.array([0]*len(job_emb) + [1]*len(cv_emb))
    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(embedding_matrix)
    score = silhouette_score(embedding_matrix, cluster_labels)
    return score

# 2 evulatie scoring methode van het model
# davis bouldin score zegt hoe goed de clusters van elkaar gescheiden zijn.-> meer gescheiden = slecht , meer compacter = goed
def davis_bouldin_scores(embedding_matrix: np.ndarray):
    n_clusters = 2
    kmeans= KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(embedding_matrix)
    score = sklearn.metrics.davies_bouldin_score(embedding_matrix,labels)
    return score

#3 evaluatie scoring methode van het model
# calinski harabasz score zegt hoe goed de clusters van elkaar gescheiden zijn.->
def calinski_harabasz_scores(embedding_matrix: np.ndarray):
    n_clusters = 2
    kmeans= KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(embedding_matrix)
    score = sklearn.metrics.calinski_harabasz_score(embedding_matrix,labels)
    return score

jobs_df = pd.read_csv("./data/processed/job_descriptions2_cleaned.csv")   # columns: job_id, job_text
cvs_df  = pd.read_csv("./data/processed/Resume_cleaned.csv")    # columns: cv_id, cv_text

#queries = dict(zip(jobs_df.job_id.astype(str), jobs_df.job_text))
#corpus  = dict(zip(cvs_df.cv_id.astype(str), cvs_df.cv_text))

job_texts = jobs_df["Job Description"].tolist()
cv_texts  = cvs_df["Resume_str"].tolist()

model_paths = ["all-MiniLM-L6-v2", "./models/tsdae_model2", "./models/tsdae_model3"]

# --- Loop over models ---
for model_path in model_paths:
    print(f"\nEvaluating model: {model_path}")
    
    model = SentenceTransformer(model_path)
    
    job_emb = model.encode(job_texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True)
    cv_emb  = model.encode(cv_texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True)
    
    all_embeddings = np.vstack([job_emb, cv_emb])
    
    sil_score = silhouette_scores(all_embeddings)
    db_score  = davis_bouldin_scores(all_embeddings)
    ch_score  = calinski_harabasz_scores(all_embeddings)
    
    print("Silhouette Score:", sil_score)
    print("Davies-Bouldin Score:", db_score)
    print("Calinski-Harabasz Score:", ch_score)