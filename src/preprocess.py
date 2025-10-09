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

class TextPreprocessing:
    def __init__(self, lemmatization=True):
        """Initialize the TextPreprocessing class with necessary NLP tools."""
        nltk.download('stopwords', quiet=True) #download once
        nltk.download('wordnet', quiet=True) #download once
        nltk.download('punkt_tab', quiet=True) #download once

        self.nlp = spacy.load("en_core_web_sm") #spacy model
        self.stop_words = set(stopwords.words('english')) #for now english stopwords only

        self.use_lemmatizer = lemmatization
        if self.use_lemmatizer:
            self.processor = WordNetLemmatizer() #uses lemmatizer
        else:
            self.processor = SnowballStemmer("english") #uses stemmer

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', str(text)) #remove extra spaces
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text) #remove special characters
        text = re.sub(r'(.)\1{2,}', r'\1\1', text) #remove character repetitions
        text = text.strip() #lowercase and strip
        return text

    def tokenize_and_stem(self, text):
        tokens = word_tokenize(text) 
        tokens = [t for t in tokens if t not in self.stop_words and t.isalpha()] #stopwords removal and keep only alphabetic tokens
        
        if self.use_lemmatizer:
            processed = [self.processor.lemmatize(t) for t in tokens] #lemmatization
        else:
            processed = [self.processor.stem(t) for t in tokens] #stemming
        return processed

    def preprocess_dataframe(self, df, columns):
        print(f"Starting preprocessing dataframe with {len(df)} rows and columns: {columns}")
        df_copy = df.copy()

        if 'Job ID' in df.columns: 
            df_copy = df.drop_duplicates(subset=['Job ID']).copy() 

        else: df_copy = df.drop_duplicates().copy()

        print(f"Dropped {len(df) - len(df_copy)} duplicates\n")

        df_copy["data"] = df_copy[columns].astype(str).agg(" ".join, axis=1) #merge columns into one string
        print("Merged columns")

        df_copy["data"] = df_copy["data"].apply(self.clean_text) #clean text
        print("Cleaned text")

        df_copy["tokens"] = df_copy["data"].apply(self.tokenize_and_stem) #tokenize and stem/lemmatize
        print("Tokenization + lemmatization/stemming done\n")

        df_copy.drop(['data'], axis=1, inplace=True) #drop intermediate column

        return df_copy
    
def main():
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)

    parser = argparse.ArgumentParser(description="Preprocess CSV datasets") #command line arguments
    parser.add_argument('--input', type=str, default='data/raw', help="Input folder path") #input folder path
    parser.add_argument('--output', type=str, default='data/processed', help="Output folder path") #output folder path
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    
    preprocessor = TextPreprocessing(lemmatization=True) 

    #Job descriptions
    input_jobs = os.path.join(input_dir, "job_descriptions2.csv") #input file path
    df_jobs = pd.read_csv(input_jobs)
    columns_to_process = ['Job Description', 'Preferred Skills', 'Work Location 1', 'Business Title'] #columns to merge and process
    processed_jobs_df = preprocessor.preprocess_dataframe(df_jobs, columns_to_process)
    output_file = os.path.join(output_dir, "job_descriptions_processed.csv")
    processed_jobs_df.to_csv(output_file, index=False)
    print(f"Saved processed Job Descriptions file to {output_file}")

    #Resumes
    input_resumes = os.path.join(input_dir, "Resume.csv") #input file path
    df_resumes = pd.read_csv(input_resumes)
    columns_to_process = ['Resume_str', 'Category'] #columns to merge and process
    processed_resumes_df = preprocessor.preprocess_dataframe(df_resumes, columns_to_process)
    output_file = os.path.join(output_dir, "resumes_processed.csv")
    processed_resumes_df.to_csv(output_file, index=False)
    print(f"Saved processed Resumes file to {output_file}")

if __name__ == "__main__":
    main()
