import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# -------------------- Logging setup --------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------- Corpus generator --------------------
class JobCorpus:
    """Lazy generator for Word2Vec training."""
    def __init__(self, resume_df, job_df):
        self.resume_df = resume_df
        self.job_df = job_df

    def __iter__(self):
        for text in self.resume_df['Resume_str']:
            yield str(text).split()
        for text in self.job_df['Job Description']:
            yield str(text).split()

# -------------------- Weighted Word2Vec --------------------
def weighted_avg_word2vec(df, text_column, w2v_model, tfidf_vectorizer):
    embeddings = []
    idf_dict = {word: tfidf_vectorizer.idf_[idx] for word, idx in tfidf_vectorizer.vocabulary_.items()}

    for text in df[text_column]:
        words = str(text).split()
        vecs, weights = [], []
        for w in words:
            if w in w2v_model.wv.key_to_index:
                vecs.append(torch.tensor(w2v_model.wv[w]))
                weights.append(idf_dict.get(w, 1.0))
        if vecs:
            vecs = torch.stack(vecs)
            weights = torch.tensor(weights, dtype=torch.float32)
            weighted_vec = (vecs * weights.unsqueeze(1)).sum(dim=0) / weights.sum()
        else:
            weighted_vec = torch.zeros(w2v_model.vector_size)
        embeddings.append(weighted_vec)
    return torch.stack(embeddings)

def embed_in_chunks(df, text_column, w2v_model, tfidf_vectorizer, chunk_size=10000):
    embeddings_list = []
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start+chunk_size]
        chunk_emb = weighted_avg_word2vec(chunk, text_column, w2v_model, tfidf_vectorizer)
        embeddings_list.append(chunk_emb)
        logging.info(f"Processed rows {start} to {start+len(chunk)}")
    return torch.cat(embeddings_list)

# -------------------- Clustering words --------------------
def cluster_words(w2v_model, threshold=0.8, topn=5):
    clusters = {}
    for word in w2v_model.wv.key_to_index:
        for similar_word, similarity in w2v_model.wv.most_similar(word, topn=topn):
            if similarity > threshold:
                cluster1 = clusters.get(word, set())
                cluster2 = clusters.get(similar_word, set())
                joined = cluster1.union(cluster2, {word, similar_word})
                for w in joined:
                    clusters[w] = joined
    # Deduplicate clusters
    unique_clusters = []
    seen = set()
    for cluster in clusters.values():
        frozen = frozenset(cluster)
        if frozen not in seen:
            unique_clusters.append(cluster)
            seen.add(frozen)
    logging.info(f"Formed {len(unique_clusters)} clusters of related words.")
    return unique_clusters

# -------------------- Ranker --------------------
class Word2VecRanker:
    """Ranker using precomputed Word2Vec embeddings for jobs and resumes."""
    def __init__(self, job_embeddings, resume_embeddings):
        self.job_embeddings = F.normalize(job_embeddings.float(), dim=1)
        self.resume_embeddings = F.normalize(resume_embeddings.float(), dim=1)

    def rank_all_jobs(self, top_k=3):
        sim_matrix = torch.mm(self.job_embeddings, self.resume_embeddings.T)
        rankings = []
        for job_id, sims in enumerate(sim_matrix):
            sims = sims.tolist()
            sorted_ids = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
            rankings.append({
                "job_id": job_id,
                "top": [(rid, sims[rid]) for rid in sorted_ids[:top_k]],
                "bottom": [(rid, sims[rid]) for rid in sorted_ids[-top_k:]]
            })
        return rankings

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Job-Resume Matching with TF-IDF Weighted Word2Vec")
    parser.add_argument('--cv', type=str, required=True, help='Path to CV dataset (CSV)')
    parser.add_argument('--jd', type=str, required=True, help='Path to Job dataset (CSV)')
    parser.add_argument('--verbose', action='store_true', help='Print sample output')
    args = parser.parse_args()

    # Load datasets
    logging.info("Loading datasets...")
    resume_df = pd.read_csv(args.cv)
    job_df = pd.read_csv(args.jd)
    logging.info(f"Loaded {len(resume_df)} resumes and {len(job_df)} job descriptions")

    # ---------------- Word2Vec Training ----------------
    logging.info("Training Word2Vec model...")
    corpus = JobCorpus(resume_df, job_df.sample(n=min(100000, len(job_df)), random_state=42))  # sample for speed
    w2v_model = Word2Vec(sentences=corpus, vector_size=50, window=20, min_count=1, workers=4, epochs=50, seed=42, sg=1)
    logging.info("Word2Vec training finished.")

    # ---- Save the model ----
    w2v_model.save("word2vec_cbow.model")
    logging.info("Word2Vec model saved to word2vec_cbow.model")

    # ---- Cluster similar words ----
    clusters = cluster_words(w2v_model, threshold=0.85)
    logging.info(f"Sample clusters: {clusters[:5]}")

    # ---------------- TF-IDF ----------------
    logging.info("Fitting TF-IDF...")
    tfidf_sample = pd.concat([resume_df['Resume_str'], job_df.sample(n=min(100000, len(job_df)), random_state=42)])
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(tfidf_sample)
    logging.info("TF-IDF fit finished.")

    # ---------------- Compute embeddings ----------------
    logging.info("Computing resume embeddings...")
    resume_embeddings = weighted_avg_word2vec(resume_df, 'Resume_str', w2v_model, tfidf_vectorizer)
    
    logging.info("Computing job embeddings in chunks...")
    job_embeddings = embed_in_chunks(job_df, 'Job Description', w2v_model, tfidf_vectorizer, chunk_size=10000)

    # ---------------- Rank ----------------
    logging.info("Ranking jobs and resumes...")
    ranker = Word2VecRanker(job_embeddings, resume_embeddings)
    rankings = ranker.rank_all_jobs(top_k=3)

    # ---------------- Prepare output ----------------
    rows = []
    for ranking in rankings:
        job_id = ranking['job_id']
        job_category = job_df.loc[job_id, 'Civil Service Title'] if 'Civil Service Title' in job_df.columns else f"Job_{job_id}"
        for rid, sim in ranking['top']:
            resume_category = resume_df.loc[rid, 'Category'] if 'Category' in resume_df.columns else f"Resume_{rid}"
            rows.append({
                'job_id': job_id,
                'job_category': job_category,
                'resume_id': rid,
                'resume_category': resume_category,
                'similarity_score': sim,
                'rank': 'top'
            })
        for rid, sim in ranking['bottom']:
            resume_category = resume_df.loc[rid, 'Category'] if 'Category' in resume_df.columns else f"Resume_{rid}"
            rows.append({
                'job_id': job_id,
                'job_category': job_category,
                'resume_id': rid,
                'resume_category': resume_category,
                'similarity_score': sim,
                'rank': 'bottom'
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv("word2vec_tfidf_rankings2edatasetlmao32223.csv", index=False)
    logging.info("Ranking finished → results saved in word2vec_tfidf_rankings.csv")

    if args.verbose:
        print(df_out.head(10))

if __name__ == "__main__":
    main()
