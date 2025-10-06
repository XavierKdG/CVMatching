import os
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from termcolor import colored
import plotly.graph_objects as go
import re

class CVJobEvaluator:
    def __init__(self, model_path):
        print(f"Loading model from {model_path}...")
        self.model = SentenceTransformer(model_path)

    @staticmethod
    def preprocess_text(text):
        text = text.lower()
        text = re.sub('[^a-z]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = ' '.join(text.split())
        return text

    def compute_similarity(self, text1, text2):
        emb1 = self.model.encode(self.preprocess_text(text1), convert_to_tensor=True)
        emb2 = self.model.encode(self.preprocess_text(text2), convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2).item() * 100
        return similarity

    @staticmethod
    def visualize_similarity(similarity):
        fig = go.Figure(go.Indicator(
            domain={'x': [0, 1], 'y': [0, 1]},
            value=similarity,
            mode="gauge+number",
            title={'text': "Matching percentage (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 50], 'color': "#FFB6C1"},
                    {'range': [50, 70], 'color': "#FFFFE0"},
                    {'range': [70, 100], 'color': "#90EE90"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 100}
            }
        ))
        fig.update_layout(width=600, height=400)
        fig.show()

    @staticmethod
    def notify(similarity):
        if similarity < 50:
            print(colored("Low chance, need to modify your CV!", "red", attrs=["bold"]))
        elif similarity < 70:
            print(colored("Good chance but you can improve further!", "yellow", attrs=["bold"]))
        else:
            print(colored("Excellent! You can submit your CV.", "green", attrs=["bold"]))

    def evaluate_dataframe(self, df, cv_column, jd_column):
        results = []
        for _, row in df.iterrows():
            cv_text = row[cv_column]
            jd_text = row[jd_column]
            similarity = self.compute_similarity(cv_text, jd_text)
            results.append(similarity)
        df['similarity'] = results
        return df

def main():
    parser = argparse.ArgumentParser(description="Evaluate CVs against Job Descriptions")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained SentenceTransformer model")
    parser.add_argument('--data_path', type=str, required=True, help="Path to processed CSV file containing CVs and JDs")
    args = parser.parse_args()

    evaluator = CVJobEvaluator(args.model_path)

    df = pd.read_csv(args.data_path)
    if 'CV' not in df.columns or 'Job Description' not in df.columns:
        raise ValueError("Processed CSV must have 'CV' and 'Job Description' columns")

    df = evaluator.evaluate_dataframe(df, cv_column='CV', jd_column='Job Description')

    # Show average similarity for quick overview
    avg_similarity = df['similarity'].mean()
    print(f"Average similarity across all CVs: {avg_similarity:.2f}%")

    evaluator.visualize_similarity(avg_similarity)
    evaluator.notify(avg_similarity)

    os.makedirs('results', exist_ok=True)
    output_file = os.path.join('results', 'evaluation_results.csv')
    df.to_csv(output_file, index=False)
    print(f"Saved detailed results to {output_file}")

if __name__ == "__main__":
    main()
