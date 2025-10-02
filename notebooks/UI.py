import streamlit as st

st.title("CV Matching UI")
cv_input = st.text_area("Plak je CV hier:")
job_choice = st.selectbox("Kies een vacature", df_jobs["Job Title"])

if st.button("Check match"):
    job_skills = df_jobs.loc[df_jobs["Job Title"] == job_choice, "skills"].values[0]
    label = hybrid_label(cv_input, job_choice, job_skills)
    st.write("Match!" if label == 1 else "Geen match.")
