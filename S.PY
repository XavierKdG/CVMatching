import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

## Case omschrijving
# Cv's kan rangschikken op basis van relevantie functie
# 2 datasets -> Job description2  = Job Postings van bedrijven  /// Job description = Zijn de Cv's van de mensen 

#Opsplitsen van tekst
# Regex over clean tekst... - > woorden splitsen van deelwoorden en werkwoordne ? onderwerp?? -> stemming / lemmatization
# stemming en lemmatization 

## Gedachteprocess
# Week 1.
#  Opzetten VM Omgeving met packages / Documentatie
#  Agency splitten van labels in verschillende secties (Business , law , Ict?)
#  column maken score van 0-1?
#  hard skills

# Week 1/2
# 1. Relevante kolommen -> Experience, job Descriptions?
# 2. Kolommen schoonmaken Voornamelijk Job Description is Raw Text (Regex of andere type.)
# ---------------------------
# 4. Label maken ->  nieuwe kolom. --> Welke CV matched nou goed
nltk.download("stopwords")  # only needed once
stop_words_nltk = set(stopwords.words("english"))

COLUMNS_TO_CLEAN = ["job description", "minimum qual requirement", "preferred skills","residency requirement"]

"""_summary_
krijgt Raw text als Arg
Return opgeschoonde tekst (zonder tekens etc..)
"""
def clean_text(text):
    # lowercase
    text = text.lower()
    # remove special chars -> keep only letters, numbers, spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # remove stopwords
    words = [w for w in text.split() if w not in stop_words_nltk]
    return " ".join(words)


def regexs (df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if any(keyword in col.lower() for keyword in COLUMNS_TO_CLEAN):  # check column name
            print(f"Cleaning column: {col}")            
            df[col] = df[col].astype(str).apply(lambda x: clean_text(x))
    return df


df = pd.read_csv("data/raw/job_descriptions.csv",sep=',')
df2 = pd.read_csv("data/raw/job_descriptions2.csv",sep=',')

regexs(df2)
#for col in df.columns:
#    print(f"{col}: {df[col].iloc[0]}")

print('\n \n \n')
for col in df2.columns:
    print(f"{col}: {df2[col].iloc[0]}")



