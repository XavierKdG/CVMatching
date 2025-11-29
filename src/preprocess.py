import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from utils import load_config

class Preprocessor:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # self.model_name = SentenceTransformer(model_name)
        self.config = load_config("configs/config.yml")

    def remove_unnecessary_columns(self, df):
        existing_cols = [col for col in self.config['job_description']['columns_to_remove'] if col in df.columns]
        return df.drop(columns=existing_cols, errors='ignore')
    
    def remove_duplicates(self, df):
        return df.drop_duplicates()
    
    def remove_html_xml_tags(self, text):
        if pd.isna(text):
            return text
        return re.sub(r'<[^>]+>', '', str(text))
    
    def keep_alphanumeric_only(self, text):
        if pd.isna(text):
            return text
        return re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
    
    def remove_urls(self, text):
        if pd.isna(text):
            return text
        return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', str(text))
    
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
        print(f"Cleaned data saved to {filepath}")
    
    def preprocess(self, df):
        df = self.remove_unnecessary_columns(df)
        df = self.remove_duplicates(df)
        df = self.apply_text_cleaning(df)
        return df

if __name__ == "__main__":
    preprocessor = Preprocessor()
    df = pd.read_csv('data/raw/job_descriptions2.csv')
    df_clean = preprocessor.preprocess(df)
    preprocessor.save_cleaned_data(df_clean, 'data/processed/job_descriptions_cleaned.csv')
        
    # resume_df = pd.read_csv('data/raw/resume.csv')
    # resume_clean = preprocessor.preprocess(resume_df)
    # preprocessor.save_cleaned_data(resume_clean, 'data/processed/resume_cleaned.csv')