from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib
import os

#data inladen
X = pd.read_csv("../data/processed/tfidf_matrix.csv")
y = pd.read_csv("../data/processed/labeled_datatest.csv")["label"]

print("X shape:", X.shape)
print("y shape:", y.shape)

#train/test split
X_train, X_test, y_train, y_test = train_test_split(      
    X, y, test_size=0.2, random_state=42
)

#model trainen
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#voorspellingen
y_pred = model.predict(X_test)

#evaluatie
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#model opslaan
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/tfidf.pkl")
print("opgeslagen in models/tfidf.pkl")
