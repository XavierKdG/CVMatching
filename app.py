# app.py
import streamlit as st
import pandas as pd
import PyPDF2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(page_title="CV Matching", layout="wide")

# --- Functie: PDF uitlezen ---
def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except:
        return ""

# --- Functie: Voorspelling ---
def predict_cv_match(cv_text, model, vectorizer, le):
    cv_tfidf = vectorizer.transform([cv_text])
    pred_probs = model.predict(cv_tfidf.toarray(), verbose=0)
    pred_class = np.argmax(pred_probs, axis=1)[0]
    label = le.inverse_transform([pred_class])[0]
    return label

# --- Laad model en preprocessing objects ---
@st.cache_resource(show_spinner=True)
def load_model_and_objects():
    model = load_model("files_chloe/models/deep_learning_model.h5")
    vectorizer = pickle.load(open("files_chloe/models/tfidf_vectorizer.pkl", "rb"))
    le = pickle.load(open("files_chloe/models/label_encoder.pkl", "rb"))
    return model, vectorizer, le

model, vectorizer, le = load_model_and_objects()

# --- Vacatures laden ---
df_jobs = pd.read_csv("data/processed/labeled_jobdescriptions2_cleaned.csv")

# --- Streamlit UI ---
st.title("CV Matching met Deep Learning")

uploaded_cv = st.file_uploader("Upload je CV (als PDF)", type=["pdf"])

if uploaded_cv:
    cv_text = extract_text_from_pdf(uploaded_cv)
    if not cv_text.strip():
        st.error("Kon geen tekst uit de PDF halen. Probeer een ander bestand.")
    else:
        if st.button("Check CV match"):
            st.info("✅ CV ontvangen, voorspelling wordt uitgevoerd...")

            predicted_label = predict_cv_match(cv_text, model, vectorizer, le)

            # --- Label 0 = geen match ---
            if predicted_label == 0:
                st.warning("❌ Geen match gevonden voor dit CV.")
            else:
                # Zoek vacatures met hetzelfde label
                matching_jobs = df_jobs[df_jobs["label"] == predicted_label]

                if not matching_jobs.empty:
                    st.success(f"✅ Match gevonden voor label '{predicted_label}'!")
                    # Laat alleen kolommen zien die bestaan (voor nu: job_text)
                    cols_to_show = [c for c in ["job_text"] if c in matching_jobs.columns]
                    st.dataframe(matching_jobs[cols_to_show])
                else:
                    st.error("❌ Geen match gevonden voor dit CV.")
