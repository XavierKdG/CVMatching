import pandas as pd
import spacy
from wai_files.text_cleaning import TextCleaner

# Load spacy model once
nlp = spacy.load("en_core_web_sm")

def extract_features(text: str, cleaner: TextCleaner) -> dict:
    """Run spaCy pipeline to extract POS, NER, and noun phrases + cleaned text."""
    doc = nlp(text)

    pos_tags = [(token.text, token.pos_) for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]

    return { 
        "clean_text": cleaner.regexs(text),   # ✅ now using the class method
        "pos_tags": pos_tags, 
        "entities": entities, 
        "noun_phrases": noun_phrases 
    }

def main():
    # Initialize cleaner
    cleaner = TextCleaner()

    # Load data
    df2 = pd.read_csv("./data/raw/job_descriptions2.csv", sep=',')

    # Clean dataframe with TextCleaner
    df_cleaned = cleaner.clean_column(df2)
    df2_features = df_cleaned.copy()

    # Apply spaCy + regex cleaning
    df2_features["features"] = df2_features["Job Description"].apply(
        lambda x: extract_features(x, cleaner)
    )

    # Unpack results
    df2_features["clean_text"]   = df2_features["features"].apply(lambda x: x["clean_text"])
    df2_features["pos_tags"]     = df2_features["features"].apply(lambda x: x["pos_tags"])
    df2_features["entities"]     = df2_features["features"].apply(lambda x: x["entities"])
    df2_features["noun_phrases"] = df2_features["features"].apply(lambda x: x["noun_phrases"])
    df2_features = df2_features.drop(columns=["features"])

    # Debug / Preview
    print("\nCleaned text:\n", df2_features["clean_text"].iloc[0]) 
    print("\nPOS Tags (first 20):\n", df2_features["pos_tags"].iloc[0][:20]) 
    print("\nNamed Entities:\n", df2_features["entities"].iloc[0]) 
    print("\nNoun Phrases:\n", df2_features["noun_phrases"].iloc[0][:15]) 

    return df2_features

if __name__ == "__main__":
    main()
