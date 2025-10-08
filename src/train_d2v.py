import os
import argparse
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from preprocess import TextPreprocessing
import yaml 
import numpy as np
import random

class Doc2VecTrainer:
    def __init__(self, config):
        self.vector_size = config.get("vector_size", 50)
        self.window = config.get("window", 5)
        self.min_count = config.get("min_count", 5)
        self.epochs = config.get("epochs", 100)
        self.alpha = config.get("alpha", 0.001)
        self.seed = config.get("seed", 42)
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
    parser.add_argument('--config', type=str, default='configs/train_config.yml', help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    input_folder = config.get("input_folder", "data/processed")
    output_folder = config.get("output_folder", "models")
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(os.path.join(input_folder, "job_descriptions_processed.csv"))
    df['tokens'] = df['tokens'].apply(eval)

    trainer = Doc2VecTrainer(config)
    tagged_data = trainer.tag_data(df)
    trainer.train_model(tagged_data)

    model_path = os.path.join(output_folder, "cv_job_matching.model")
    trainer.save_model(model_path)

    embeddings = [trainer.model.dv[str(i)].tolist() for i in range(len(df))]
    df['embeddings'] = embeddings
    embedding_csv_path = os.path.join(input_folder, "job_descriptions_embeddings.csv")
    df.to_csv(embedding_csv_path, index=False)
    print(f"Embeddings saved to {embedding_csv_path}")

if __name__ == "__main__":
    main()
