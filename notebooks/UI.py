import streamlit as st
import pandas as pd
import PyPDF2
from testcvmatching import hybrid_label #functie uit andere bestand importeren

#vacatures laden
df_jobs = pd.read_csv("../data/raw/job_descriptions.csv")
df_jobs2 = pd.read_csv("../data/raw/job_descriptions2.csv")

def extract_text_from_pdf(uploaded_file):
    #haalt tekst uit een geüploade PDF
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:   
        text += page.extract_text() or ""
    return text

st.title("CV Matching")

#upload stuk voor cv
uploaded_cv = st.file_uploader("Upload je CV (als PDF)", type=["pdf"])

#vacature kiezen
job_choice = st.selectbox("Kies een vacature", df_jobs["Job Title"])

if st.button("Check CV match"):
    if uploaded_cv is None:
        st.warning("Upload eerst een PDF van je CV.")
    else:
        cv_text = extract_text_from_pdf(uploaded_cv)
        #skills van de vacature
        job_row = df_jobs[df_jobs["Job Title"] == job_choice].iloc[0]
        job_title = job_row["Job Title"]
        job_skills = job_row["skills"]

        #geimporteerde functie gebruiken 
        label = hybrid_label(cv_text, job_title, job_skills)

        #resultaat
        if label == 1:
            st.success("Match tussen CV en vacature gevonden")
        else:
            st.error("Geen match gevonden")

