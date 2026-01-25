import argparse
import logging
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from utils import load_config, setup_logging, get_device

class Preprocessor:
    def __init__(self, config, model_name=None):
        if model_name is None:
            model_name = config['preprocessing']['model_name']
        self.config = config
        self.device = str(get_device(config))
        self.model = SentenceTransformer(model_name, device=self.device)

    def remove_unnecessary_columns(self, df, columns_to_remove) -> pd.DataFrame:
        existing_cols = [col for col in columns_to_remove if col in df.columns]
        return df.drop(columns=existing_cols, errors='ignore')

    def remove_duplicates(self, df) -> pd.DataFrame:
        return df.drop_duplicates()
    
    def remove_html_xml_tags(self, text)-> str:
        if pd.isna(text):
            return text
        return re.sub(r'<[^>]+>', '', str(text))
    
    def keep_alphanumeric_only(self, text:str)-> str:
        if pd.isna(text):
            return text
        return re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
    
    def remove_urls(self, text)-> str:
        if pd.isna(text):
            return text
        return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', str(text))
    
    def remove_extra_whitespace(self, text):
        if pd.isna(text):
            return text
        return re.sub(r'\s+', ' ', str(text)).strip()

    def clean_text(self, text):
        text = self.remove_html_xml_tags(text)
        text = self.remove_urls(text)
        text = self.keep_alphanumeric_only(text)
        text = self.remove_extra_whitespace(text)
        return text
    
    def apply_text_cleaning(self, df):
        df_cleaned = df.copy()
        for col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].apply(self.clean_text)
        return df_cleaned
    
    def save_cleaned_data(self, df, filepath):
        df.to_csv(filepath, index=False)
        logging.info(f"Cleaned data saved to {filepath}")
    
    def preprocess(self, df, columns_to_remove):
        df = self.remove_unnecessary_columns(df, columns_to_remove)
        df = self.remove_duplicates(df)
        df = self.apply_text_cleaning(df)
        return df

def process_dataset(preprocessor, section_name, config):
    """Generalized dataset loader + processor based on config."""
    raw_folder = config["paths"]["raw_folder"]
    processed_folder = config["paths"]["processed_folder"]

    section = config[section_name]
    input_file = section["input_file"]
    cols_to_remove = section["columns_to_remove"]

    raw_path = f"{raw_folder}/{input_file}"
    output_path = f"{processed_folder}/{input_file.replace('.csv', '_cleaned.csv')}"

    logging.info(f"--- Processing {section_name}: {raw_path} ---")

    df = pd.read_csv(raw_path)
    df_clean = preprocessor.preprocess(df, cols_to_remove)
    preprocessor.save_cleaned_data(df_clean, output_path)

    logging.info(f"--- Finished {section_name} ---\n")

def main(config_path=None):
    """Main function to initialize and run the preprocessor."""
    if config_path is None:
        config_path = "configs/config.yml"

    config = load_config(config_path)
    preprocessor = Preprocessor(config)

    process_dataset(preprocessor, "job_description", config)
    process_dataset(preprocessor, "resume", config)

    logging.info("--- All Preprocessing Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yml", help="Path to the config.yml file")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config["logging"]["file_name"])

    device = get_device(config)
    logging.info(f"Preprocessing running on device: {device}")

    main(config_path=args.config)   