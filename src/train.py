import os
import argparse
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from datetime import datetime
import random
import logging
import ast
from .utils import setup_logging, load_config #help functions

class Doc2VecTrainer:
    """Class responsible for training and saving Doc2Vec models."""
    def __init__(self, config, config_path=None):
        training_cfg  = config['training']

        self.vector_size = training_cfg.get("vector_size", 50)
        self.window = training_cfg.get("window", 5)
        self.min_count = training_cfg.get("min_count", 5)
        self.epochs = training_cfg.get("epochs", 100)
        self.alpha = training_cfg.get("alpha", 0.001)
        self.seed = training_cfg.get("seed", 42)
        
        base_model_folder = config['paths'].get("model_folder", "models")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = os.path.join(base_model_folder, f"run_{timestamp}")
        os.makedirs(self.output_folder, exist_ok=True)

        config_name = os.path.splitext(os.path.basename(config_path or "config.yml"))[0]
        self.model_name = f"cv_job_matching_{config_name}.model"

        random.seed(self.seed)
        np.random.seed(self.seed)

        logging.info(f"Trainer initialized for {self.epochs} epochs and vector size {self.vector_size}.")

    def tag_data(self, df, prefix=''):
        tagged_data = []

        for i, row in df.iterrows():
            words = list(row['tokens']) if 'tokens' in row and row['tokens'] is not None else []

            if 'entities' in df.columns and pd.notna(row['entities']):
                entities_dict = {}
                if isinstance(row['entities'], dict):
                    entities_dict = row['entities']
                elif isinstance(row['entities'], str):
                    try:
                        entities_dict = ast.literal_eval(row['entities'])
                    except Exception:
                        import json
                        try:
                            entities_dict = json.loads(row['entities'])
                        except Exception as e:
                            logging.warning(f"Could not parse entities for row {i}: {e}")

                for ent_type, ent_values in entities_dict.items():
                    if ent_values is not None:
                        if isinstance(ent_values, (list, tuple, np.ndarray)):
                            iterable_values = ent_values
                        else:
                            iterable_values = [ent_values]

                        for val in iterable_values:
                            val_clean = "_".join(str(val).split())
                            words.append(f"{ent_type}_{val_clean}")

            tagged_data.append(TaggedDocument(words=words, tags=[f"{prefix}_{i}"]))

        return tagged_data

    def train_model(self, tagged_data):
        """Train a Doc2Vec model on tagged data."""
        self.model = Doc2Vec(vector_size=self.vector_size,
                             window=self.window,
                             min_count=self.min_count,
                             epochs=self.epochs,
                             alpha=self.alpha,
                             seed=self.seed)
        
        self.model.build_vocab(tagged_data)

        logging.info(f'Starting training for {self.epochs} epochs...')
        self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        logging.info('Training completed')
        return self.model

    def save_model(self):
        """Save the trained Doc2Vec model."""
        model_path = os.path.join(self.output_folder, self.model_name)
        self.model.save(model_path)
        logging.info(f"Model saved at {model_path}")
        return model_path
    
    def generate_embeddings(self, df, prefix):
        """Genereert embeddings op basis van de DataFrame-index."""
        logging.info(f"Generating embeddings for {len(df)} documents with prefix '{prefix}'...")
        embeddings = [self.model.dv[f"{prefix}_{i}"].tolist() for i in df.index]
        
        df["embeddings"] = embeddings
        logging.info(f"Embeddings generated successfully for prefix '{prefix}'.")
        return df
    
def parse_arguments():
    """Parse command-line arguments to override the config."""
    parser = argparse.ArgumentParser(description="Train Doc2Vec model on job descriptions and resumes")
    parser.add_argument('--config', type=str, default='configs/config.yml', help="Path to YAML config file")
    args = parser.parse_args()
    logging.info(f"Arguments parsed: {args}")
    return args

def load_datasets(config):
    """Load and preprocess datasets from configured paths."""
    processed_folder = config["paths"]["processed_folder"]

    jobs_file = os.path.splitext(config["datasets"]["jobs"]["input_filename"])[0] + "_processed.parquet"
    resumes_file = os.path.splitext(config["datasets"]["resumes"]["input_filename"])[0] + "_processed.parquet"

    jobs_path = os.path.join(processed_folder, jobs_file)
    resumes_path = os.path.join(processed_folder, resumes_file)

    logging.info(f"Loading job descriptions from: {jobs_path}")
    logging.info(f"Loading resumes from: {resumes_path}")

    jobs_df = pd.read_parquet(jobs_path)
    resumes_df = pd.read_parquet(resumes_path)

    return jobs_df, resumes_df

def save_embeddings(df, config, dataset_key):
    """Save dataframe with embeddings to processed folder."""
    processed_folder = config["paths"]["processed_folder"]

    input_filename = config["datasets"][dataset_key]["input_filename"]
    base_name = os.path.splitext(input_filename)[0]
    output_filename = f"{base_name}_embeddings.parquet"

    output_path = os.path.join(processed_folder, output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)

    logging.info(f"{dataset_key.capitalize()} embeddings saved to {output_path}")

def main(config_path=None): 
    """Main function to preprocess datasets."""
    args = parse_arguments()
    config_file = config_path or args.config #parse command line arguments
    config = load_config(config_path or args.config) #load config

    jobs_df, resumes_df = load_datasets(config) #read job/resume dfs
    trainer = Doc2VecTrainer(config, config_path=config_file) #configure class

    tagged_jobs = trainer.tag_data(jobs_df, "job")
    tagged_resumes = trainer.tag_data(resumes_df, "cv")
    all_tagged = tagged_jobs + tagged_resumes
    
    trainer.train_model(all_tagged) #train model
    trainer.save_model() #save model

    logging.info("Generating embeddings for jobs...")
    jobs_df = trainer.generate_embeddings(jobs_df, "job") #create jobs embeddings

    logging.info("Generating embeddings for resumes...")
    resumes_df = trainer.generate_embeddings(resumes_df, "cv") #create resumes embeddings

    save_embeddings(jobs_df, config, "jobs") #save job embeddings 
    save_embeddings(resumes_df, config, "resumes") #save resume embeddings 

    logging.info("--- Training and embedding generation completed succesfully. ---")

if __name__ == "__main__":
    setup_logging() #setup logging
    main()
