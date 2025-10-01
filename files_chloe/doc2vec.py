import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


class Doc2VecMatcher:
    def __init__(self, vector_size=100, window=5, min_count=2, epochs=40):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None

    def prepare_documents(self, df: pd.DataFrame):
        tagged_docs = []
        for _, row in df.iterrows():
            tagged_docs.append(
                TaggedDocument(
                    words=row['resume_text'].lower().split(),
                    tags=[f"resume_{row['resume_id']}"]
                )
            )
            tagged_docs.append(
                TaggedDocument(
                    words=row['job_text'].lower().split(),
                    tags=[f"job_{row['job_id']}"]
                )
            )
        return tagged_docs

    def train(self, tagged_docs):
        self.model = Doc2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            epochs=self.epochs
        )
        self.model.build_vocab(tagged_docs)
        self.model.train(tagged_docs, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        return self.model

    def add_vectors_to_df(self, df: pd.DataFrame):
        resume_vecs = []
        job_vecs = []
        sims = []

        for _, row in df.iterrows():
            resume_id = f"resume_{row['resume_id']}"
            job_id = f"job_{row['job_id']}"
            if resume_id in self.model.dv and job_id in self.model.dv:
                r_vec = self.model.dv[resume_id]
                j_vec = self.model.dv[job_id]
                sim = cosine_similarity([r_vec], [j_vec])[0][0]
            else:
                r_vec = np.zeros(self.vector_size)
                j_vec = np.zeros(self.vector_size)
                sim = 0.0

            resume_vecs.append(r_vec)
            job_vecs.append(j_vec)
            sims.append(sim)

        df['resume_vec'] = resume_vecs
        df['job_vec'] = job_vecs
        df['similarity'] = sims
        return df


def main():
    df = pd.read_csv("/home/admin-groep11/CVMatching-1/data/processed/labeled_jobdescriptions2.csv")
    df = df.sample(2000, random_state=42)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    matcher = Doc2VecMatcher(vector_size=50, window=5, min_count=2, epochs=30)
    tagged_docs = matcher.prepare_documents(train_df)
    matcher.train(tagged_docs)

    train_df = matcher.add_vectors_to_df(train_df)
    test_df = matcher.add_vectors_to_df(test_df)

    y_true = test_df['label']
    y_pred = (test_df['similarity'] > 0.5).astype(int)

    print("\n=== Evaluatie (Doc2Vec cosine similarity) ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    return test_df


if __name__ == "__main__":
    results = main()
    print(results.head())
