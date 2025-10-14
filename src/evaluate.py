import os
import argparse
import pandas as pd
import numpy as np
import re
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import TextPreprocessing
import PyPDF2
from sklearn.preprocessing import normalize

class ResumeEvaluator:
    def __init__(self, model_path):
        self.model = Doc2Vec.load(model_path)
        self.preprocessor = TextPreprocessing(lemmatization=True)

    def preprocess_text(self, text):
        text = re.sub(r'[^a-zA-Z ]', ' ', text)
        text = ' '.join(text.lower().split())
        tokens = self.preprocessor.tokenize_and_stem(text)
        return tokens

    def infer_vector(self, text):
        tokens = self.preprocess_text(text)
        return self.model.infer_vector(tokens)

    def read_pdf(self, pdf_path):
        pdf = PyPDF2.PdfReader(pdf_path)
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    
def main():
    parser = argparse.ArgumentParser(description="Evaluate CVs against Job Descriptions")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained Doc2Vec model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to processed data folder")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    jobs_path = os.path.join(args.data_path, "job_descriptions_processed.csv")
    resumes_path = os.path.join(args.data_path, "resumes_processed.csv")

    jobs_df = pd.read_csv(jobs_path)
    resumes_df = pd.read_csv(resumes_path)

    jobs_df['tokens'] = jobs_df['tokens'].apply(eval)
    resumes_df['tokens'] = resumes_df['tokens'].apply(eval)

    evaluator = ResumeEvaluator(args.model_path)

    jobs_vectors = []
    for i in range(len(jobs_df)):
        tag = f"job_{i}"
        if tag in evaluator.model.dv:
            jobs_vectors.append(evaluator.model.dv[tag])
        else:
            raise KeyError(f"Tag {tag} not found in model.dv")
    jobs_vectors = normalize(np.stack(jobs_vectors))

    resumes_vectors = []
    for i in range(len(resumes_df)):
        tag = f"cv_{i}"
        if tag in evaluator.model.dv:
            resumes_vectors.append(evaluator.model.dv[tag])
        else:
            raise KeyError(f"Tag {tag} not found in model.dv")
    resumes_vectors = normalize(np.stack(resumes_vectors))

    similarity_matrix = cosine_similarity(resumes_vectors, jobs_vectors)
    similarity_matrix = ((similarity_matrix + 1) / 2) * 100


    results = []
    for i, resume_row in resumes_df.iterrows():
        for j, job_row in jobs_df.iterrows():
            results.append({
                "Resume_ID": resume_row.get("ID", i),
                "Resume_Category": resume_row.get("Category", ""),
                "Job_ID": job_row.get("Job ID", j),
                "Job_Category": job_row.get("Job Category", ""),
                "Similarity_%": round(similarity_matrix[i, j], 2)
            })

    results_df = pd.DataFrame(results)
    results_file = os.path.join("results", "resume_job_similarity.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Similarity results saved to {results_file}")

if __name__ == "__main__":
    main()
