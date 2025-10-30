import streamlit as st
import PyPDF2 
import re
import os
import yaml
import numpy as np
from src.evaluate import ResumeEvaluator 

st.set_page_config(page_title="Resume Ranker", page_icon="🤭", layout="centered")

@st.cache_resource
def load_config(config_path="configs/config.yml"):
    """Loads the YAML configuration."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Config file not found at {config_path}: {e}")
        return None

@st.cache_resource
def load_evaluator(config):
    """
    Loads the ResumeEvaluator class, which contains the similarity model.
    This is cached so the model only loads once.
    """
    st.info("Cache: Loading Resume Evaluator (and similarity model)...")
    try:
        return ResumeEvaluator(config)
    except Exception as e:
        st.error(f"Error loading evaluator: {e}")
        return None

def extract_pdf_text(pdf_file):
    """Reads text from an uploaded PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def clean_text(text):
    """Cleaning function (identical to preprocess.py)."""
    text = re.sub(r'\s+', ' ', str(text))
    text = re.sub(r'(.)\1{2,}', r'\1\1', text) 
    text = re.sub(r'http\S+|www\.\S+', '', text)
    return text.strip().lower()

def main():
    st.title("Resume Ranker")
    st.write("Upload one job description and multiple resumes to rank the candidates.")

    config = load_config()
    if not config: st.stop()

    evaluator = load_evaluator(config)
    if not evaluator:
        st.error("Evaluator could not be loaded. Check logs.")
        st.stop()
    
    st.success("Evaluator loaded successfully.")
    st.divider()

    jd_text_input = st.text_area("1. Paste the Job Description here", height=200)
    
    uploaded_resumes = st.file_uploader("2. Upload all candidate resumes (PDF)", type="pdf", accept_multiple_files=True)

    if uploaded_resumes and jd_text_input:
        
        with st.spinner(f"Ranking {len(uploaded_resumes)} resumes..."):
            
            results_list = []

            jd_text = clean_text(jd_text_input)

            for cv_file in uploaded_resumes:
                resume_text = clean_text(extract_pdf_text(cv_file))
                
                if not resume_text:
                    st.warning(f"Could not read {cv_file.name}. Skipping.")
                    continue

                result_data = evaluator.evaluate_match(resume_text, jd_text)

                results_list.append({
                    "filename": cv_file.name,
                    "score": result_data["total_score"],
                    "semantic": result_data["semantic_score"],
                    "keyword": result_data["keyword_score"],
                    "required_skills": result_data["required_skills"],
                    "overlap": result_data["overlapping_skills"]
                })
        
        if results_list:

            sorted_results = sorted(results_list, key=lambda x: x['score'], reverse=True)
            
            st.header(f"Results", divider="rainbow")
            
            st.subheader("Top 3 Candidates")
            cols = st.columns(min(len(sorted_results), 3))
            for i, result in enumerate(sorted_results[:3]):
                with cols[i]:
                    st.metric(
                        label=f"Rank #{i+1}: {result['filename']}",
                        value=f"{result['score']*100:.1f}%"
                    )
            
            st.divider()
            st.subheader("All Candidates (Ranked)")

            for i, result in enumerate(sorted_results):
                
                with st.container(border=True):
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader(f"Rank #{i+1}: {result['filename']}")
                    
                    with col2:
                        st.metric(label="Total Match", value=f"{result['score']*100:.1f}%")

                    with st.expander("View Breakdown & Keyword Analysis"):
                        st.subheader("Score Breakdown")
                        col1_exp, col2_exp = st.columns(2)
                        col1_exp.metric(label="Semantic Match (60%)", 
                                    value=f"{result['semantic']*100:.1f}%")
                                    
                        col2_exp.metric(label="Keyword Match (40%)", 
                                    value=f"{result['keyword']*100:.1f}%")
                        
                        st.subheader("Keyword Analysis")
                        st.write(f"**Required keywords (found in job description):** `{len(result['required_skills'])}`")
                        st.write(result['required_skills'] if result['required_skills'] else "None")
                        st.write(f"**Found in resume (overlap):** `{len(result['overlap'])}`")
                        st.write(result['overlap'] if result['overlap'] else "None")
                st.empty() 

if __name__ == "__main__":
    main()