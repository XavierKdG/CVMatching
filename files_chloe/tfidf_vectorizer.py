import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("data/processed/labeled_jobdescriptions2_cleaned.csv")

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))
vectorizer.fit(df["job_text"])

le = LabelEncoder()
le.fit(df["label"])

pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))
