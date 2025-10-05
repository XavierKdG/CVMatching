import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer om resume_text en job_text apart te vectoriseren
class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.key]

# === 1. Laad gelabelde dataset ===
df = pd.read_csv("/home/admin-groep11/CVMatching-1/data/processed/labeled_jobdescriptions2.csv")

# === 2. Train/test split ===
X = df[['resume_text', 'job_text']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === 3. Vectorizers voor resume en job teksten ===
resume_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
job_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

features = FeatureUnion([
    ("resume", resume_vectorizer),
    ("job", job_vectorizer)
], n_jobs=-1)

# === 4. Train feature matrices ===
X_train_features = features.fit_transform(X_train.to_dict('records'))
X_test_features = features.transform(X_test.to_dict('records'))

# === 5. Random Forest model ===
rf = RandomForestClassifier(
    n_estimators=200, 
    max_depth=None, 
    random_state=42, 
    n_jobs=-1
)

rf.fit(X_train_features, y_train)
y_pred = rf.predict(X_test_features)

# === 6. Evaluatie ===
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
