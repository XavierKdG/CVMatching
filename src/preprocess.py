import pandas as pd
import os 
import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import argparse
import logging
from .utils import setup_logging, load_config #help functions

class TextPreprocessing:
    def __init__(self, use_lemmatization=True, use_ner=False):
        """Initialize the TextPreprocessing class with necessary NLP tools."""
        logging.info("Initialiseren van TextPreprocessor...")
        nltk.download('stopwords', quiet=True) #download once
        nltk.download('wordnet', quiet=True) #download once
        nltk.download('punkt_tab', quiet=True) #download once

        self.nlp = spacy.load("en_core_web_lg") #spacy model
        self.stop_words = set(stopwords.words('english')) #for now english stopwords only

        self.use_lemmatizer = use_lemmatization
        self.processor = WordNetLemmatizer() if use_lemmatization else SnowballStemmer("english") #choose lemmatizer or stemmer

        self.use_ner = use_ner #use NER

        processor_type = "Lemmatizer" if use_lemmatization else "Stemmer" #for logging
        logging.info(f"Preprocessor initialized with {processor_type}.") 
        if self.use_ner:
            logging.info("Named Entity Recognition (NER) ENABLED.")

    def clean_text(self, text):
        """Clean the input text by removing unwanted characters and formatting."""
        text = re.sub(r'\s+', ' ', str(text)) #remove extra spaces
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text) #remove special characters
        text = re.sub(r'(.)\1{2,}', r'\1\1', text) #remove character repetitions
        text = re.sub(r'http\S+|www\.\S+', '', text) #remove http & www URLs (used AI for this regex)
        return text.strip() #remove leading/trailing spaces
    
    def extract_entities(self, texts):
        """Batch NER extraction using spaCy pipe."""
        entities_list = []
        for doc in self.nlp.pipe(texts, batch_size=50): 
            entities = {}
            for ent in doc.ents:
                entities.setdefault(ent.label_, []).append(ent.text)

            entities = {k: list(set(v)) for k, v in entities.items()}
            entities_list.append(entities)
        return entities_list

    def tokenize_and_process(self, text):
        """Tokenize the text and apply stemming or lemmatization."""
        tokens = word_tokenize(text) 
        tokens = [t for t in tokens if t not in self.stop_words and t.isalpha()] #stopwords removal and keep only alphabetic tokens
        
        if self.use_lemmatizer:
            processed = [self.processor.lemmatize(t) for t in tokens] #lemmatization
        else:
            processed = [self.processor.stem(t) for t in tokens] #stemming
        return processed

    def process_dataframe(self, df, columns_to_combine, duplicate_subset=None):
        """Process the specified columns of the dataframe."""
        logging.info(f"Processing dataframe with {len(df)} rows.")
        df_copy = df.copy() 
        initial_rows = len(df_copy)

        if duplicate_subset and all(col in df_copy.columns for col in duplicate_subset):
            df_copy.drop_duplicates(subset=duplicate_subset, inplace=True)
        else: 
            df_copy.drop_duplicates(inplace=True)
        logging.info(f"{initial_rows - len(df_copy)} duplicate rows removed")

        df_copy["combined_text"] = df_copy[columns_to_combine].astype(str).agg(" ".join, axis=1) #merge columns into one string
        df_copy["cleaned_text"] = df_copy["combined_text"].apply(self.clean_text) #clean text
        df_copy["tokens"] = df_copy["cleaned_text"].apply(self.tokenize_and_process) #tokenize and stem/lemmatize

        if self.use_ner:
            logging.info("Extracting Named Entities in batch...")
            df_copy["entities"] = self.extract_entities(df_copy["cleaned_text"].tolist())

        return df_copy.drop(columns=['combined_text', 'cleaned_text']) #drop intermediate columns
    
def process_file(preprocessor, dataset_config, raw_dir, processed_dir, chunksize=50000):
    """Process a dataset (CSV) in chunks and save as Parquet."""
    file_name = dataset_config["input_filename"]
    input_path = os.path.join(raw_dir, file_name)
    output_filename = os.path.splitext(file_name)[0] + "_processed.parquet"
    output_path = os.path.join(processed_dir, output_filename)

    logging.info(f"--- Processing file: {file_name} with chunksize={chunksize} ---")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    processed_chunks = []
    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        processed_df = preprocessor.process_dataframe(
            df=chunk,
            columns_to_combine=dataset_config["columns_to_process"],
            duplicate_subset=dataset_config.get("duplicate_subset")
        )
        processed_chunks.append(processed_df)

    pd.concat(processed_chunks).to_parquet(output_path, index=False)
    logging.info(f"File saved to {output_path}")

def main(input_dir=None, output_dir=None, config_path=None, parse_args=True):
    """
    Preprocess datasets.
    - input_dir/output_dir: optional overrides
    - config_path: optional config file
    - parse_args: set False when called from pipeline
    """
    config = load_config(config_path) #load config

    if parse_args:
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", type=str, default=config["paths"]["raw_folder"])
        parser.add_argument("--output", type=str, default=config["paths"]["processed_folder"])
        parser.add_argument("--config", type=str, default=config_path or "configs/config.yml")
        args = parser.parse_args()
        input_dir = args.input
        output_dir = args.output
        config_path = args.config

    input_dir = input_dir or config["paths"]["raw_folder"]
    output_dir = output_dir or config["paths"]["processed_folder"]
    chunksize = config["preprocessing"].get("chunksize", 50000)

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    preprocessor = TextPreprocessing(
        use_lemmatization=config['preprocessing']['use_lemmatization'], #initialize preprocessor
        use_ner=config['preprocessing']['use_ner']) #initialize NER

    process_file(preprocessor, config["datasets"]["jobs"], input_dir, output_dir, chunksize=chunksize)
    process_file(preprocessor, config["datasets"]["resumes"], input_dir, output_dir, chunksize=chunksize)

    logging.info("--- Preprocessing complete ---")

if __name__ == "__main__":
    setup_logging() #setup logging
    main()
