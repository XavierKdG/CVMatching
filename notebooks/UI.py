import streamlit as st
import pandas as pd
import PyPDF2

# Voorbeeld vacatures (vervang dit door je echte dataset)
df_jobs = pd.DataFrame({
    "Job Title": ["Data Scientist", "Software Engineer"],
    "skills": ["Python, ML, SQL", "Python, React, Docker"]
})

def extract_text_from_pdf(uploaded_file):
    """Haalt tekst uit een geüploade PDF."""
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def hybrid_label(cv_text, job_title, job_skills):
    # Dummy matching functie: vervang dit door je eigen ML/logica
    return 1 if job_skills.lower() in cv_text.lower() else 0

st.title("📄 CV Matching UI")

# PDF upload in plaats van text area
uploaded_cv = st.file_uploader("Upload je CV (PDF)", type=["pdf"])

job_choice = st.selectbox("Kies een vacature", df_jobs["Job Title"])

if st.button("Check match"):
    if uploaded_cv is None:
        st.warning("⚠️ Upload eerst een PDF van je CV.")
    else:
        cv_text = extract_text_from_pdf(uploaded_cv)
        job_skills = df_jobs.loc[df_jobs["Job Title"] == job_choice, "skills"].values[0]
        label = hybrid_label(cv_text, job_choice, job_skills)
        st.write("✅ Match!" if label == 1 else "❌ Geen match.")
