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
import yaml

class TextPreprocessing:
    def __init__(self, lemmatization=True):
        """Initialize the TextPreprocessing class with necessary NLP tools."""
        logging.info("Initialiseren van TextPreprocessor...")
        nltk.download('stopwords', quiet=True) #download once
        nltk.download('wordnet', quiet=True) #download once
        nltk.download('punkt_tab', quiet=True) #download once

        self.nlp = spacy.load("en_core_web_sm") #spacy model
        self.stop_words = set(stopwords.words('english')) #for now english stopwords only

        self.use_lemmatizer = lemmatization
        self.processor = WordNetLemmatizer() if lemmatization else SnowballStemmer("english") #choose lemmatizer or stemmer
        processor_type = "Lemmatizer" if lemmatization else "Stemmer" #for logging
        logging.info(f"Preprocessor initialized with {processor_type}.") 

    def clean_text(self, text):
        """Clean the input text by removing unwanted characters and formatting."""
        text = re.sub(r'\s+', ' ', str(text)) #remove extra spaces
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text) #remove special characters
        text = re.sub(r'(.)\1{2,}', r'\1\1', text) #remove character repetitions
        text = re.sub(r'http\S+|www\.\S+', '', text) #remove http & www URLs (used AI for this regex)
        return text.strip() #remove leading/trailing spaces
 
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

        return df_copy.drop(columns=['combined_text', 'cleaned_text']) #drop intermediate columns
    
def setup_logging():
    """Configure logging to write to a file."""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='logs/preprocessing.log',
        filemode='w')
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info("Logging is configured.")

def parse_arguments(config):
    """Parse command-line arguments to override the config."""
    parser = argparse.ArgumentParser(description="Preprocess datasets, with config and overrides.")
    parser.add_argument('--input', type=str, default=config['data']['raw_folder'], help="Override the input folder.")
    parser.add_argument('--output', type=str, default=config['data']['processed_folder'], help="Override the output folder.")
    args = parser.parse_args()
    logging.info(f"Arguments parsed: {args}")
    return args
    
def load_config(config_path="configs/config.yml"):
    """Load configuration from a YAML file."""
    logging.info(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def process_file(preprocessor, config_section, raw_dir, processed_dir):
    """Process a single CSV file based on the configuration."""
    file_name = config_section['input_file']
    input_path = os.path.join(raw_dir, file_name)
    output_path = os.path.join(processed_dir, config_section['output_file'])

    logging.info(f"--- Processing file: {file_name} ---")
    try:
        df = pd.read_csv(input_path)
        processed_df = preprocessor.process_dataframe(df=df, columns_to_combine=config_section['columns_to_process'], duplicate_subset=config_section.get('duplicate_subset'))
        processed_df.to_csv(output_path, index=False)
        logging.info(f"File saved to {output_path}")
    except FileNotFoundError:
        logging.error(f"File not found: {input_path}")
    except Exception as e:
        logging.error(f"An error occurred while processing {file_name}: {e}")

def main():
    """Main function to preprocess datasets."""
    setup_logging() #setup logging
    
    config = load_config() #load config
    args = parse_arguments(config) #parse command line arguments

    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)

    preprocessor = TextPreprocessing(lemmatization=config['preprocessing']['lemmatization']) #initialize preprocessor

    process_file(preprocessor, config['preprocessing']['jobs'], args.input, args.output)
    process_file(preprocessor, config['preprocessing']['resumes'], args.input, args.output)

    logging.info("--- Preprocessing complete ---")

if __name__ == "__main__":
    main()
