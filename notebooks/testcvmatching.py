#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[ ]:


df1 = pd.read_csv('../data/raw/job_descriptions.csv')
df2 = pd.read_csv('../data/raw/job_descriptions2.csv')
df1.head()


# In[3]:


df2 = pd.read_csv('../data/raw/job_descriptions.csv')
df2.head()


# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def rule_based_label(cv_text, job_title, job_skills):
    "Label = 1 als er overlap is tussen cv_text en de job skills of title."

    keywords = str(job_title).lower().split() + str(job_skills).lower().split(",")
    for kw in keywords:
        if kw.strip() in cv_text.lower():
            return 1
    return 0

def similarity_label(cv_text, job_text, threshold=0.5):

    "Bereken TF-IDF cosine geljkenis tussen cv en vacature."

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([cv_text, job_text])
    sim = cosine_similarity(tfidf[0], tfidf[1])[0][0]
    return 1 if sim >= threshold else 0

def hybrid_label(cv_text, job_title, job_skills, threshold=0.5):
    job_text = f"{job_title} {job_skills}"
    rb = rule_based_label(cv_text, job_title, job_skills)
    sim = similarity_label(cv_text, job_text, threshold)
    return 1 if (rb == 1 or sim == 1) else 0

if __name__ == "__main__":
    #laad vacatures
    df_jobs = pd.read_csv('../data/raw/job_descriptions.csv')

    #simuleer CV’s
    cvs = [
        "I am a data scientist skilled in Python and machine learning.",
        "Experienced marketing specialist with focus on SEO and campaigns."
    ]

    results = []
    for cv in cvs:
        for idx, row in df_jobs.iterrows():
            job_title = row["Job Title"]
            job_skills = row["skills"]

            label = hybrid_label(cv, job_title, job_skills)

            results.append({
                "cv_text": cv,
                "job_title": job_title,
                "job_skills": job_skills,
                "label": label
            })

    #opslaan van gelabelde dataset
    df_out = pd.DataFrame(results)
    df_out.to_csv("data/processed/labeled_datatest.csv", index=False)
    print("opgeslagen in data/processed/labeled_data.csv")
    print(df_out.head())

