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
    def __init__(self, model_name='all-MiniLM-L6-v2', lemmatization=True, embed_model=True):
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

        self.embed_model = embed_model
        if self.embed_model:
            self.bert_model = SentenceTransformer(model_name) #BERT model

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', str(text)) #remove extra spaces
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text) #remove special characters
        text = text.lower().strip() #lowercase and strip
        return text

    def tokenize_and_stem(self, text):
        tokens = word_tokenize(text) 
        tokens = [t for t in tokens if t.lower() not in self.stop_words and t.isalpha()] #stopwords removal and keep only alphabetic tokens
        
        if self.use_lemmatizer:
            print("Using lemmatization")
            processed = [self.processor.lemmatize(t) for t in tokens] #lemmatization
        else:
            print("Using stemming")
            processed = [self.processor.stem(t) for t in tokens] #stemming
        return processed

    def preprocess_dataframe(self, df, columns, embed_text=True):
        df_copy = df.drop_duplicates().copy() #drop duplicates

        df_copy["data"] = df_copy[columns].astype(str).agg(" ".join, axis=1) #merge columns into one string
        df_copy["data"] = df_copy["data"].apply(self.clean_text) #clean text
        df_copy["tokens"] = df_copy["data"].apply(self.tokenize_and_stem) #tokenize and stem/lemmatize

        if self.embed_model and embed_text:
            df_copy["embeddings"] = list(self.bert_model.encode(df_copy["data"].tolist())) #BERT embeddings

        return df_copy[["data", "tokens"] + (["embeddings"] if self.embed_model and embed_text else [])]

def load_raw_csv(file_path):
    return pd.read_csv(file_path)

def main():
    # df1 = pd.read_csv("./data/raw/job_descriptions.csv")
    df2 = pd.read_csv("./data/raw/job_descriptions2.csv")

    preprocessor = TextPreprocessing(model_name='all-MiniLM-L6-v2', lemmatization=True)
    print("preprocessor initialized")

    # # preproccessing df1
    # df1_cols = ['Job Description', 'Skills', 'Experience', 'Responsibilities']
    # df1_processed = preprocessor.preprocess_dataframe(df1, df1_cols)
    # df1_processed[["data", "tokens"]].to_csv('./data/processed/df1_processed.csv', index=False)

    # preproccessing df2
    df2_cols = ['Job Description', 'Preferred Skills'] #columns to be merged and processed
    df2_processed = preprocessor.preprocess_dataframe(df2, df2_cols) 
    df2_processed[["data", "tokens"]].to_csv('./data/processed/df2_processed.csv', index=False)

if __name__ == "__main__":
    main()
