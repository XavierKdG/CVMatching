import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from text_cleaning import TextCleaner


nlp = spacy.load("en_core_web_sm")

"""Run spaCy pipeline to extract POS, NER, and noun phrases.""" 
def extract_features(text: str) -> dict: 
    doc = nlp(text)

    pos_tags = [(token.text, token.pos_) for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]

    return { 
        "clean_text": regexs(text), 
        "pos_tags": pos_tags, 
        "entities": entities, 
        "noun_phrases": noun_phrases }



cleaner = TextCleaner()
df2 = pd.read_csv("./data/raw/job_descriptions2.csv",sep=',')
df2_features = df2.copy()

df_cleaned = cleaner.clean_dataframe(df2)

df2_features["features"] = df2_features["Job Description"].apply(extract_features)

df2_features["clean_text"] = df2_features["features"].apply(lambda x: x["clean_text"])
df2_features["pos_tags"] = df2_features["features"].apply(lambda x: x["pos_tags"])
df2_features["entities"] = df2_features["features"].apply(lambda x: x["entities"])
df2_features["noun_phrases"] = df2_features["features"].apply(lambda x: x["noun_phrases"])

df2_features = df2_features.drop(columns=["features"])

print("\nCleaned text:\n", df2_features["clean_text"].iloc[0]) 
print("\nPOS Tags (first 20):\n", df2_features["pos_tags"].iloc[0][:20]) 
print("\nNamed Entities:\n", df2_features["entities"].iloc[0]) 
print("\nNoun Phrases:\n", df2_features["noun_phrases"].iloc[0][:15])