# preprocessing.py
import re
import pandas as pd
import spacy
import time

class TextPreprocessor:
    def __init__(self, text_columns=None, remove_stopwords=True, remove_numbers=True,
                 remove_special_chars=True, lowercase=True, sample_size=10555, random_state=42):
        print("🟢 Initialiseren van TextPreprocessor...")
        self.text_columns = text_columns or ["resume_text", "Job Description"]
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers
        self.remove_special_chars = remove_special_chars
        self.lowercase = lowercase
        self.sample_size = sample_size
        self.random_state = random_state

        print("📦 Laden van spaCy-model (en_core_web_sm)...")
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        print("✅ SpaCy-model geladen.\n")

    def _basic_cleaning(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        if self.lowercase:
            text = text.lower()
        if self.remove_numbers:
            text = re.sub(r"\d+", " ", text)
        if self.remove_special_chars:
            text = re.sub(r"[^a-zA-Z\s.,!?]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _lemmatize_text(self, text: str) -> str:
        text = self._basic_cleaning(text)
        doc = self.nlp(text)
        if self.remove_stopwords:
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        else:
            tokens = [token.lemma_ for token in doc if not token.is_punct]
        return " ".join(tokens)

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        print("🧹 Cleaning dataframe gestart...\n")
        df = df.copy()

        if self.sample_size is not None and self.sample_size < len(df):
            print(f"✂️  Sampling dataset: {self.sample_size} van {len(df)} rijen (random).")
            df = df.sample(n=self.sample_size, random_state=self.random_state).reset_index(drop=True)
        else:
            print("ℹ️  Geen sampling toegepast (volledige dataset wordt gebruikt).")

        for col in self.text_columns:
            if col not in df.columns:
                print(f"⚠️  Kolom '{col}' niet gevonden — overslaan.")
                continue

            print(f"➡️  Start cleaning voor kolom: {col}")
            cleaned = []
            for i, text in enumerate(df[col]):
                if i % 100 == 0:
                    print(f"   🔄 Bezig met rij {i}/{len(df)} in kolom '{col}'...")
                cleaned_text = self._lemmatize_text(str(text))
                cleaned.append(cleaned_text)
            df[col] = cleaned
            print(f"✅ Kolom '{col}' klaar ({len(df)} rijen verwerkt)\n")
            time.sleep(0.5)

        before = len(df)
        existing_cols = [c for c in self.text_columns if c in df.columns]
        if existing_cols:
            df = df.drop_duplicates(subset=existing_cols, keep="first").reset_index(drop=True)
        else:
            print("⚠️  Geen van de opgegeven text_columns gevonden — overslaan duplicatenverwijdering.")
        after = len(df)
        print(f"♻️  Duplicaten verwijderd: {before - after}")
        print("🏁 Cleaning voltooid.\n")

        return df

# run_preprocessing.py
from preprocessing import TextPreprocessor
import pandas as pd

# === 1. Laad je dataset ===
df = pd.read_csv("/home/admin-groep11/CVMatching-1/data/processed/labeled_jobdescriptions2.csv")

# === 2. Initialiseer de preprocessor ===
preprocessor = TextPreprocessor(
    text_columns=["resume_text", "Job Description"],
    remove_stopwords=True,
    remove_numbers=True,
    remove_special_chars=True,
    lowercase=True
)

# === 3. Start de cleaning ===
df_cleaned = preprocessor.clean_dataframe(df)

# === 4. Opslaan van het resultaat ===
output_path = "/home/admin-groep11/CVMatching-1/data/processed/labeled_jobdescriptions2_cleaned.csv"
df_cleaned.to_csv(output_path, index=False, encoding="utf-8")

print(f"✅ Schoongemaakte dataset opgeslagen naar:\n{output_path}")
