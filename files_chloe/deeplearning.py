import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# === 1. Laad dataset ===
df = pd.read_csv("/home/admin-groep11/CVMatching-1/data/processed/labeled_jobdescriptions2_cleaned.csv")

X = df["job_text"]
y = df["label"]

# === 2. Encode labels naar integers ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === 3. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# === 4. TF-IDF vectorizer laden (of trainen indien nog niet gedaan) ===
# Optie A: laden vanuit pickle bestand
#with open("/home/admin-groep11/CVMatching-1/data/processed/tfidf_vectorizer.pkl", "rb") as f:
    #vectorizer = pickle.load(f)

# Optie B: opnieuw trainen op trainingsdata (als pickle nog niet bestaat)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Transformeer de data
X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# === 5. Bouw het deep learning model ===
model = Sequential([
    Dense(256, input_dim=X_train_tfidf.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')  # aantal classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === 6. Early stopping ===
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# === 7. Train het model ===
history = model.fit(
    X_train_tfidf.toarray(),  # Keras verwacht dense arrays
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=2
)

# === 8. Evalueer op testset ===
# === 8. Evaluatie ===
y_pred_probs = model.predict(X_test_tfidf.toarray(), verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# === 9. Bereken scores ===
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# === 10. Print resultaten ===
print("\n=== Model Scores ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"MSE:       {mse:.4f}")
print(f"RMSE:      {rmse:.4f}")


