import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextCleaner:
    """ A class to clean job description text data.
        Includes methods to clean text, clean DataFrame columns,""" 
    
    def __init__(self, columns_to_clean=None, keep_single=None):
        nltk.download("stopwords", quiet=True)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.columns_to_clean = [c.lower().strip() for c in (
            columns_to_clean or [
                "job description",
                "minimum qual requirement",
                "preferred skills",
            ]
        )]
        self.keep_single = keep_single or {"a", "i"}

    def regexs(self, text: str) -> str:
        """Cleans a single text string using regex: lowercases, removes special characters, using regex.
        it's a helper function for clean_column.
        input = text string
        output = cleaned text string"""

        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)   # remove special chars
        text = re.sub(r"\s+", " ", text).strip()   # normalize spaces
        words = [
            self.lemmatizer.lemmatize(w)
            for w in text.split()
            if w not in self.stop_words
        ]
        words = [w for w in words if len(w) > 1 or w in self.keep_single]
        return " ".join(words) 

    def clean_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans specified text columns in a DataFrame
        input = pandas dataframe
        output = pandas dataframes cleaned ."""

        for col in df.columns:
            if any(keyword in col.lower() for keyword in self.columns_to_clean):
                print(f"Cleaning column: {col}")
                df[col] = df[col].astype(str).apply(self.regexs)
        return df
    
    def remove_duplicates(self, df: pd.DataFrame, id_column="Job Id") -> pd.DataFrame:
        """Removes duplicate rows based on a specified ID column.
        input = pandas dataframe, id_column name (default "Job Id")
        output = pandas dataframe without duplicates."""

        if id_column in df.columns:
            df = df.drop_duplicates(subset=[id_column]).reset_index(drop=True)
            print(f" Removed duplicates '{id_column}'")
        else:
            print(f" Column '{id_column}' not found")
        return df


    def process_file(self, input_path: str, output_path: str, sep: str):
        """Processes a CSV file: loads, cleans, removes duplicates, and saves.
        input = input file path, output file path, separator for the csv
        output = cleaned pandas dataframe."""

        print(f"{input_path}")
        df = pd.read_csv(input_path, sep=sep)
        df = self.clean_column(df)
        df = self.remove_duplicates(df, id_column="Job Id")
        df.to_csv(output_path, index=False)
        print(f"saved at '{output_path}'")
        return df


def main():
    cleaner = TextCleaner()
    df_cleaned = cleaner.process_file(
        input_path="./data/raw/job_descriptions.csv",
        output_path="./data/processed/job_descriptions_cleaned2.csv",
        sep=','
    )
    return df_cleaned
        

if __name__ == "__main__":
    main()
