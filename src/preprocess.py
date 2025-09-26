import pandas as pd
import os 
import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)

# Nog toe te voegen misschien:
# elongation, negation handling, spellchecker

class TextPreprocessing:
    def __init__(self, model_name='all-MiniLM-L6-v2', lemmatization=True):
        """Initialize the TextPreprocessing class with necessary NLP tools."""
        nltk.download('stopwords', quiet=True) #download once
        nltk.download('wordnet', quiet=True) #download once

        self.nlp = spacy.load("en_core_web_sm") #spacy model
        self.stop_words = set(stopwords.words('english')) #for now english stopwords only

        self.use_lemmatizer = lemmatization
        if self.use_lemmatizer:
            self.processor = WordNetLemmatizer() #uses lemmatizer
        else:
            self.processor = SnowballStemmer("english") #uses stemmer

        self.bert_model = SentenceTransformer(model_name) #BERT model 

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', str(text)) #remove extra spaces
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text) #remove special characters
        text = text.lower().strip() #lowercase and strip
        return text
    
    def preprocess_dataframe(self, df, columns):
        df_copy = df.drop_duplicates().copy() #remove duplicates
        for col in columns:
            if col in df_copy.columns:
                df_copy[col] = self.preprocess_texts(df_copy[col])
            else:
                print(f"Column '{col}' does not exist in the dataset.")
        return df_copy
    
    def tokenize_and_stem(self, text):
        doc = self.nlp(text)
        tokens = [token.text for token in doc if token.text.lower() not in self.stop_words]
        
        if self.use_lemmatizer:
            processed = [self.processor.lemmatize(token) for token in tokens]
        else:
            processed = [self.processor.stem(token) for token in tokens]
        return processed
    
    def preprocess_texts(self, texts):
        return [self.tokenize_and_stem(self.clean_text(t)) for t in texts]
    
    def encode_texts(self, texts):
        return self.bert_model.encode(texts, convert_to_tensor=True)
    

def load_raw_csv(file_path):
    return pd.read_csv(file_path)

def main():
    # df1 = pd.read_csv("./data/raw/job_descriptions.csv")
    df2 = pd.read_csv("./data/raw/job_descriptions2.csv")

    preprocessor = TextPreprocessing()

    # # preproccessing df1
    # df1_columns_to_process = ['Job Description', 'Skills', 'Experience', 'Responsibilities']
    # df1_processed = preprocessor.preprocess_dataframe(df1, df1_columns_to_process)
    # df1_processed.to_csv('./data/processed/processed_df1.csv', index=False)

    # preproccessing df2
    df2_columns_to_process = ['Job Description', 'Preferred Skills']
    df2_processed = preprocessor.preprocess_dataframe(df2, df2_columns_to_process)
    df2_processed.to_csv('./data/processed/processed_df2.csv', index=False)

if __name__ == "__main__":
    main()


