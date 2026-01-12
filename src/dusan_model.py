"""
End-to-end CV ↔ Job matching pipeline

This script:
- Preprocesses job descriptions and resumes according to specification
- Creates embeddings using all-MiniLM-L6-v2
- Matches resumes to jobs using cosine similarity

====================
USAGE EXAMPLE
====================

In PowerShell:
python cvmatching_pipeline.py \
    --jobs data/job_descriptions2.csv \
    --resumes data/Resume.csv \
    --job_text_col "Job Description" \
    --resume_text_col "Resume_str" \
    --top_n 5 \
    --output matches.csv

Or in CMD:
python src\dusan_model.py --jobs data\raw\job_descriptions2.csv --resumes data\raw\Resume.csv --job_text_col "Job Description" --resume_text_col "Resume_str" --top_n 5 --output data\processed\matches.csv

====================
OUTPUT
====================
A CSV file containing:
- resume_id
- job_id
- similarity_score
"""

import argparse
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch


# ----------------------
# Preprocessing
# ----------------------

def preprocess_text(text: str) -> str:
    """Clean text according to the provided rules."""
    if pd.isna(text) or str(text).strip() == "":
        return "NO DESCRIPTION"

    # Remove HTML / XML tags
    try:
        text = BeautifulSoup(str(text), "lxml").get_text(separator=" ")
    except Exception:
        # Fallback if lxml is not installed
        text = BeautifulSoup(str(text), "html.parser").get_text(separator=" ")

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Keep only alphanumeric tokens (preserve capitalization)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)

    # Remove extra whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ----------------------
# Column removal helpers
# ----------------------

JOB_DROP_COLS = [
    "Posting Type", "# Of Positions", "Title Code No", "Level",
    "Job Category", "Full-Time/Part-Time indicator",
    "Division/Work Unit", "Additional Information", "To Apply",
    "Work Location 1", "Recruitment Contact", "Posting Date",
    "Post Until", "Posting Updated", "Process Date",
    "Salary Range From", "Salary Range To", "Salary Frequency",
    "Work Location", "Hours/Shift", "Residency Requirement",
]

RESUME_DROP_COLS = [
    "Resume_html",
]


# ----------------------
# Main pipeline
# ----------------------

def main(args):
    # Load data
    jobs_df = pd.read_csv(args.jobs)
    resumes_df = pd.read_csv(args.resumes)

    # Drop unnecessary columns
    jobs_df = jobs_df.drop(columns=[c for c in JOB_DROP_COLS if c in jobs_df.columns])
    resumes_df = resumes_df.drop(columns=[c for c in RESUME_DROP_COLS if c in resumes_df.columns], errors="ignore")

    # Remove duplicates
    jobs_df = jobs_df.drop_duplicates()
    resumes_df = resumes_df.drop_duplicates()

    # Preprocess text columns
    jobs_df["clean_text"] = jobs_df[args.job_text_col].apply(preprocess_text)
    resumes_df["clean_text"] = resumes_df[args.resume_text_col].apply(preprocess_text)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    # Create embeddings
    job_embeddings = model.encode(
        jobs_df["clean_text"].tolist(),
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    resume_embeddings = model.encode(
        resumes_df["clean_text"].tolist(),
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Similarity
    similarity_matrix = cosine_similarity(resume_embeddings, job_embeddings)

    # Top-N matches per resume
    results = []
    for resume_idx, sims in enumerate(similarity_matrix):
        top_indices = np.argsort(sims)[::-1][: args.top_n]
        for job_idx in top_indices:
            results.append(
                {
                    "resume_id": resume_idx,
                    "job_id": job_idx,
                    "similarity_score": float(sims[job_idx]),
                }
            )

    matches_df = pd.DataFrame(results)

    # Save output
    matches_df.to_csv(args.output, index=False)
    print(f"Saved matches to: {args.output}")


# ----------------------
# CLI
# ----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV ↔ Job matching pipeline")
    parser.add_argument("--jobs", required=True, help="Path to job descriptions CSV")
    parser.add_argument("--resumes", required=True, help="Path to resumes CSV")
    parser.add_argument("--job_text_col", required=True, help="Text column name in jobs CSV")
    parser.add_argument("--resume_text_col", required=True, help="Text column name in resumes CSV")
    parser.add_argument("--top_n", type=int, default=5, help="Top-N matches per resume")
    parser.add_argument("--output", default="matches.csv", help="Output CSV path")

    main(parser.parse_args())
