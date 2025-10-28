import streamlit as st
import pandas as pd
import os
from files_chloe.tfidf_vectorizer import get_vectorizer
from files_chloe.model import get_model
from sklearn.metrics.pairwise import cosine_similarity
import re

# ======== Functie om tekst schoon te maken ========
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# ======== Laad vectorizer, model en vacatures ========
@st.cache_resource
def load_resources():
    vectorizer = get_vectorizer()
    model = get_model()
    jobs_path = os.path.join("data", "processed", "labeled_jobdescriptions2_cleaned.csv")
    jobs = pd.read_csv(jobs_path)
    jobs["job_text"] = jobs["job_text"].apply(clean_text)
    return vectorizer, model, jobs

vectorizer, model, jobs = load_resources()

# ======== Streamlit UI ========
st.title("CV–Vacature Matching Tool")
st.write("Upload een CV-bestand om de beste match te vinden in de vacatures.")

uploaded_file = st.file_uploader("Upload je CV (TXT of PDF)", type=["txt", "pdf"])

if uploaded_file is not None:
    import io
    from PyPDF2 import PdfReader

    # CV tekst uitlezen
    if uploaded_file.type == "application/pdf":
        pdf = PdfReader(uploaded_file)
        cv_text = "\n".join([page.extract_text() for page in pdf.pages])
    else:
        cv_text = uploaded_file.read().decode("utf-8")

    cv_text = clean_text(cv_text)
    st.subheader("CV Inhoud (eerste 500 tekens):")
    st.write(cv_text[:500] + "...")

    # Vectoriseer CV en vacatures
    cv_vec = vectorizer.transform([cv_text])
    job_vecs = vectorizer.transform(jobs["job_text"])

    # Cosine similarity
    sims = cosine_similarity(cv_vec, job_vecs)[0]
    jobs["similarity"] = sims

    # Top 5 matches
    top_matches = jobs.sort_values(by="similarity", ascending=False).head(5)

    st.subheader("Top 5 meest vergelijkbare vacatures:")
    for i, row in top_matches.iterrows():
        st.write(f"Score: {row['similarity']:.3f}")
        st.write(row["job_text"][:300] + "...")
        st.markdown("---")
