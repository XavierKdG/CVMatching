import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer om een kolom te selecteren
class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.key]

# === 1. Dataset ===
df = pd.read_csv("/home/admin-groep11/CVMatching-1/data/processed/labeled_jobdescriptions2_cleaned.csv")

# Check even kolommen
print("Kolommen:", df.columns.tolist())

# === 2. Train/test split ===
X = df[['resume_text', 'job_text']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 3. Maak pipelines voor elke tekstkolom ===
resume_pipeline = Pipeline([
    ('selector', TextSelector(key='resume_text')),
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2)))
])

job_pipeline = Pipeline([
    ('selector', TextSelector(key='job_text')),
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2)))
])

# === 4. Combineer beide vectorizers ===
combined_features = FeatureUnion([
    ('resume', resume_pipeline),
    ('job', job_pipeline)
])

# === 5. Bouw het volledige model ===
model = Pipeline([
    ('features', combined_features),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])

# === 6. Train het model ===
print("🚀 Model wordt getraind...")
model.fit(X_train, y_train)

# === 7. Voorspel en evalueer ===
y_pred = model.predict(X_test)
print("\n✅ Evaluatie:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print(classification_report(y_test, y_pred))
