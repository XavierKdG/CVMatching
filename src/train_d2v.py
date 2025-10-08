import os
import argparse
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from preprocess import TextPreprocessing
import pyyaml 
import numpy as np
import random

class Doc2VecTrainer:
    def __init__(self, vector_size=50, window=5, min_count=5, epochs=100, alpha=0.001, seed=42):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.alpha = alpha
        self.seed = seed
        self.model = None

        random.seed(self.seed)
        np.random.seed(self.seed)

    def tag_data(self, df):
        tagged_data = [TaggedDocument(words=row['tokens'], tags=[str(i)]) for i, row in df.iterrows()]
        return tagged_data

    def train_model(self, tagged_data):
        self.model = Doc2Vec(vector_size=self.vector_size,
                             window=self.window,
                             min_count=self.min_count,
                             epochs=self.epochs,
                             alpha=self.alpha,
                             seed=self.seed)
        self.model.build_vocab(tagged_data)
        for epoch in range(self.epochs):
            print(f"Training epoch {epoch+1}/{self.epochs}")
            self.model.train(tagged_data, total_examples=len(tagged_data), epochs=1)
        return self.model

    def save_model(self, output_path):
        self.model.save(output_path)
        print(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Doc2Vec model on job descriptions")
    parser.add_argument('--input', type=str, default='data/processed', help="Processed data folder")
    parser.add_argument('--output', type=str, default='models', help="Output folder for model")
    parser.add_argument('--vector_size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load processed data
    df = pd.read_csv(os.path.join(args.input, "job_descriptions_processed.csv"))
    
    # Ensure 'tokens' column is list type
    df['tokens'] = df['tokens'].apply(eval)

    trainer = Doc2VecTrainer(vector_size=args.vector_size, epochs=args.epochs)
    tagged_data = trainer.tag_data(df)
    trainer.train_model(tagged_data)

    model_path = os.path.join(args.output, "cv_job_matching.model")
    trainer.save_model(model_path)


if __name__ == "__main__":
    main()
