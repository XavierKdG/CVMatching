import sys
from pathlib import Path
import streamlit as st
from gensim.models.doc2vec import Doc2Vec
from qdrant_client import QdrantClient
import pandas as pd
import re
import PyPDF2
from sklearn.preprocessing import normalize

# --- ADD PROJECT ROOT TO PATH ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.preprocess import TextPreprocessing  # Your preprocessing class

# --- PAGE CONFIG ---
st.set_page_config(page_title="CV Matcher", page_icon="📄", layout="centered")

# --- STATIC IMPORTS ---
MODEL_PATH = PROJECT_ROOT / "models/cv_job_matching_config_20251017_024323.model"
EVALUATOR_MODEL = Doc2Vec.load(str(MODEL_PATH))
PREPROCESSOR = TextPreprocessing(use_lemmatization=True)
QDRANT_CLIENT = QdrantClient(url="http://localhost:6333")

# --- HELPER FUNCTIONS ---

def read_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def preprocess_text(text):
    """Clean, lowercase, tokenize, and optionally lemmatize."""
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = " ".join(text.split())
    tokens = PREPROCESSOR.tokenize_and_process(text)
    return tokens

def infer_vector(text):
    tokens = preprocess_text(text)
    return EVALUATOR_MODEL.infer_vector(tokens)

# --- UI ---
st.title("📄 CV & Job Matcher")
st.write("Upload your CV and get the top 10 matching job descriptions!")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

if uploaded_file is not None:
    st.write("Processing your CV...")

    # Extract text from uploaded file
    if uploaded_file.type == "application/pdf":
        cv_text = read_pdf(uploaded_file)
    else:
        cv_text = str(uploaded_file.read(), "utf-8")

    # --- VECTOR INFERENCE ---
    vector = infer_vector(cv_text)

    # --- QDRANT SEARCH ---
    search_results = QDRANT_CLIENT.search(
        collection_name="job2_embeddings",
        query_vector=vector.tolist(),
        limit=10
    )

    # --- DISPLAY RESULTS ---
    st.write("Top 10 Matching Jobs:")
    for result in search_results:
        st.write(f"**Job Title:** {result.payload.get('Business Title', 'N/A')}")
        st.write(f"**Similarity Score:** {round(result.score * 100, 2)}%")
        st.divider()
