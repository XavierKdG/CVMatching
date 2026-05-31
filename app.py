import streamlit as st
import PyPDF2
import re
import os
import yaml
import numpy as np
import pandas as pd
from src.evaluate import ResumeEvaluator

st.set_page_config(page_title="Resume Ranker", page_icon="🤭", layout="wide")

@st.cache_resource
def load_config(config_path="configs/config.yml"):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Config file not found at {config_path}: {e}")
        return None

@st.cache_resource(show_spinner="Loading model...")
def load_evaluator(config, model_name):
    try:
        return ResumeEvaluator(config, model_name=model_name)
    except Exception as e:
        st.error(f"Error loading evaluator: {e}")
        return None

SIMILARITY_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "multi-qa-mpnet-base-cos-v1",
    "all-distilroberta-v1",
    "all-MiniLM-L12-v2",
    "paraphrase-MiniLM-L6-v2",
]

def extract_pdf_text(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def extract_file_text(uploaded_file):
    if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
        return extract_pdf_text(uploaded_file)
    try:
        return uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Could not read {uploaded_file.name}: {e}")
        return None

def clean_text(text):
    text = re.sub(r'\s+', ' ', str(text))
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    return text.strip().lower()

def entry_label(text, fallback_prefix, existing_count):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines:
        label = lines[0][:50]
        if len(lines[0]) > 50:
            label += "..."
        return label
    return f"{fallback_prefix} #{existing_count + 1}"

def render_entry_list(entries, label_key, remove_prefix):
    for i, entry in enumerate(list(entries)):
        col1, col2 = st.columns([5, 1])
        with col1:
            with st.expander(entry["label"]):
                st.text_area("", value=entry["text"], height=150, disabled=True, label_visibility="collapsed", key=f"text_{remove_prefix}_{i}")
        with col2:
            if st.button("Remove", key=f"{remove_prefix}_{i}", width='stretch'):
                st.session_state[label_key].pop(i)
                st.rerun()

if "jd_entries" not in st.session_state:
    st.session_state.jd_entries = []
if "resume_entries" not in st.session_state:
    st.session_state.resume_entries = []
if "jd_upload_key" not in st.session_state:
    st.session_state.jd_upload_key = 0
if "resume_upload_key" not in st.session_state:
    st.session_state.resume_upload_key = 0
if "jd_method" not in st.session_state:
    st.session_state.jd_method = "Upload file"
if "resume_method" not in st.session_state:
    st.session_state.resume_method = "Upload file"

def main():
    st.title("Resume Ranker")
    st.markdown("Add one or more job descriptions and resumes, then rank candidates by how well they match.")

    config = load_config()
    if not config:
        st.stop()

    model_name = st.selectbox("Similarity model", SIMILARITY_MODELS, index=0)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Job Descriptions")
        with st.container(border=True):
            jd_method = st.radio("", ["Paste text", "Upload file"], key="jd_method", horizontal=True, label_visibility="collapsed")

            if jd_method == "Upload file":
                jd_files = st.file_uploader("Upload job descriptions (PDF / TXT)", type=["pdf", "txt"], accept_multiple_files=True, key=f"jd_upload_{st.session_state.jd_upload_key}")
                if st.button("Add", key="add_jd_file", width='stretch') and jd_files:
                    for jd_file in jd_files:
                        text = extract_file_text(jd_file)
                        if text:
                            st.session_state.jd_entries.append({"label": jd_file.name, "text": text})
                    st.session_state.jd_upload_key += 1
                    st.rerun()
            else:
                with st.form(key="jd_paste_form", clear_on_submit=True):
                    jd_text = st.text_area("Paste job description text", height=150, label_visibility="collapsed")
                    if st.form_submit_button("Add", width='stretch') and jd_text.strip():
                        label = entry_label(jd_text, "JD", len(st.session_state.jd_entries))
                        st.session_state.jd_entries.append({"label": label, "text": jd_text})
                        st.rerun()

            if st.session_state.jd_entries:
                st.markdown("**Added job descriptions:**")
                render_entry_list(st.session_state.jd_entries, "jd_entries", "jd")
                if st.button("Clear all JDs", key="clear_jd"):
                    st.session_state.jd_entries.clear()
                    st.rerun()

    with col2:
        st.subheader("2. Resumes")
        with st.container(border=True):
            resume_method = st.radio("", ["Paste text", "Upload file"], key="resume_method", horizontal=True, label_visibility="collapsed")

            if resume_method == "Upload file":
                resume_files = st.file_uploader("Upload resumes (PDF / TXT)", type=["pdf", "txt"], accept_multiple_files=True, key=f"resume_upload_{st.session_state.resume_upload_key}")
                if st.button("Add", key="add_resume_file", width='stretch') and resume_files:
                    for resume_file in resume_files:
                        text = extract_file_text(resume_file)
                        if text:
                            st.session_state.resume_entries.append({"label": resume_file.name, "text": text})
                    st.session_state.resume_upload_key += 1
                    st.rerun()
            else:
                with st.form(key="resume_paste_form", clear_on_submit=True):
                    resume_text = st.text_area("Paste resume text", height=150, label_visibility="collapsed")
                    if st.form_submit_button("Add", width='stretch') and resume_text.strip():
                        label = entry_label(resume_text, "Resume", len(st.session_state.resume_entries))
                        st.session_state.resume_entries.append({"label": label, "text": resume_text})
                        st.rerun()

            if st.session_state.resume_entries:
                st.markdown("**Added resumes:**")
                render_entry_list(st.session_state.resume_entries, "resume_entries", "resume")
                if st.button("Clear all resumes", key="clear_resume"):
                    st.session_state.resume_entries.clear()
                    st.rerun()

    st.divider()

    # ── Rank ──
    jds = st.session_state.jd_entries
    resumes = st.session_state.resume_entries

    if jds and resumes:
        keyword_enabled = st.checkbox("Enable keyword matching (60/40 split)", value=False, help="When disabled, ranking is 100% based on semantic similarity.")
        compare_mode = st.checkbox("Compare all models", value=False, help="Run evaluation against every similarity model and compare scores side by side")
        if st.button("Rank Resumes", type="primary", width='stretch'):
            sem_weight = 1.0 if not keyword_enabled else 0.6
            kw_weight = 0.0 if not keyword_enabled else 0.4
            if compare_mode:
                models_to_run = SIMILARITY_MODELS
                with st.spinner(f"Matching {len(resumes)} resume(s) against {len(jds)} JD(s) using {len(models_to_run)} models..."):
                    for jd_entry in jds:
                        jd_text = clean_text(jd_entry["text"])
                        all_results = {}
                        for model_name in models_to_run:
                            m_evaluator = load_evaluator(config, model_name)
                            if not m_evaluator:
                                continue
                            m_evaluator.set_keyword_enabled(keyword_enabled)
                            resume_texts = [clean_text(r["text"]) for r in resumes]
                            batch_data = m_evaluator.evaluate_batch(resume_texts, jd_text)
                            all_results[model_name] = {
                                resumes[i]["label"]: batch_data[i]["total_score"]
                                for i in range(len(resumes))
                            }

                        rows = []
                        for res_entry in resumes:
                            row = {"Resume": res_entry["label"]}
                            scores = []
                            for model_name in models_to_run:
                                if model_name in all_results:
                                    s = all_results[model_name].get(res_entry["label"], 0)
                                    row[model_name] = round(s * 100, 1)
                                    scores.append(s)
                            row["Average"] = round(np.mean(scores) * 100, 1) if scores else 0
                            rows.append(row)

                        df = pd.DataFrame(rows).sort_values("Average", ascending=False).reset_index(drop=True)
                        df.insert(0, "Rank", range(1, len(df) + 1))
                        score_cols = [c for c in df.columns if c not in ("Rank", "Resume")]

                        with st.expander(f"Comparison for: {jd_entry['label']}", expanded=True):
                            styled = df.style.background_gradient(
                                cmap="RdYlGn",
                                subset=score_cols,
                                vmin=0, vmax=100
                            )
                            st.dataframe(styled, width='stretch', hide_index=True)
            else:
                evaluator = load_evaluator(config, model_name)
                if not evaluator:
                    st.stop()
                evaluator.set_keyword_enabled(keyword_enabled)
                with st.spinner(f"Matching {len(resumes)} resume(s) against {len(jds)} job description(s)..."):
                    for jd_entry in jds:
                        jd_text = clean_text(jd_entry["text"])
                        resume_texts = [clean_text(r["text"]) for r in resumes]
                        batch_data = evaluator.evaluate_batch(resume_texts, jd_text)
                        results = [
                            {
                                "filename": resumes[i]["label"],
                                "score": batch_data[i]["total_score"],
                                "semantic": batch_data[i]["semantic_score"],
                                "keyword": batch_data[i]["keyword_score"],
                                "required": batch_data[i]["required_skills"],
                                "overlap": batch_data[i]["overlapping_skills"]
                            }
                            for i in range(len(resumes))
                        ]
                        results.sort(key=lambda x: x["score"], reverse=True)

                        with st.expander(f"Results for: {jd_entry['label']}", expanded=False):
                            if results:
                                st.subheader("Top 3 Candidates")
                                cols = st.columns(min(len(results), 3))
                                for i, r in enumerate(results[:3]):
                                    with cols[i]:
                                        st.metric(label=f"#{i+1}: {r['filename']}", value=f"{r['score']*100:.1f}%")

                                st.divider()
                                st.subheader("All Candidates (Ranked)")
                                for i, r in enumerate(results):
                                    with st.container(border=True):
                                        col1, col2 = st.columns([2, 1])
                                        col1.subheader(f"Rank #{i+1}: {r['filename']}")
                                        col2.metric(label="Total Match", value=f"{r['score']*100:.1f}%")
                                        with st.expander("Score Breakdown & Keyword Analysis"):
                                            c1, c2 = st.columns(2)
                                            c1.metric(label=f"Semantic Match ({int(sem_weight*100)}%)", value=f"{r['semantic']*100:.1f}%")
                                            c2.metric(label=f"Keyword Match ({int(kw_weight*100)}%)", value=f"{r['keyword']*100:.1f}%")
                                            st.markdown(f"**Required keywords:** `{len(r['required'])}`")
                                            st.write(r["required"] if r["required"] else "None")
                                            st.markdown(f"**Found in resume:** `{len(r['overlap'])}`")
                                            st.write(r["overlap"] if r["overlap"] else "None")
                            else:
                                st.info("No valid results for this job description.")
    elif not jds and not resumes:
        st.info("Add at least one job description and one resume to start ranking.")
    else:
        st.info("Add both job descriptions and resumes, then press **Rank Resumes**.")

if __name__ == "__main__":
    main()
