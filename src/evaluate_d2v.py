# src/evaluate.py
import argparse
import re
import PyPDF2
import numpy as np
from numpy.linalg import norm
from gensim.models.doc2vec import Doc2Vec
import plotly.graph_objects as go
from preprocess import TextPreprocessing

class ResumeEvaluator:
    def __init__(self, model_path):
        self.model = Doc2Vec.load(model_path)
        self.preprocessor = TextPreprocessing(lemmatization=True)

    def preprocess_text(self, text):
        text = re.sub(r'[^a-zA-Z ]', ' ', text)
        text = ' '.join(text.lower().split())
        tokens = self.preprocessor.tokenize_and_stem(text)
        return tokens

    def read_pdf(self, pdf_path):
        pdf = PyPDF2.PdfReader(pdf_path)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text
    
