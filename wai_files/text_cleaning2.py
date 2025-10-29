# text_cleaning2.py
import pandas as pd
import re
import nltk
import argparse
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from logger import setup_logger

# 1️⃣ Setup logger
logger = setup_logger(name="GlobalLogger", log_file="./logs/all_processes.log")


class TextCleaner:
    """ A class to clean job description text data.
        Includes methods to clean text, clean DataFrame columns,
        remove duplicates, and save processed files.
    """

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
        """Cleans a single text string using regex: lowercases, removes special characters."""
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
        """Cleans specified text columns in a DataFrame."""
        for col in df.columns:
            if any(keyword in col.lower() for keyword in self.columns_to_clean):
                logger.info(f"Cleaning column: {col}")
                df[col] = df[col].astype(str).apply(self.regexs)
        return df

    def remove_duplicates(self, df: pd.DataFrame, id_column="Job ID") -> pd.DataFrame:
        """Removes duplicate rows based on a specified ID column."""
        if id_column in df.columns:
            df = df.drop_duplicates(subset=[id_column]).reset_index(drop=True)
            logger.info(f"Removed duplicates '{id_column}'")
        else:
            logger.warning(f"Column '{id_column}' not found")
        return df

    def process_file(self, input_path: str, output_path: str, sep: str = ",") -> pd.DataFrame:
        """Processes a CSV file: loads, cleans, removes duplicates, and saves."""
        logger.info(f"Loading file: {input_path}")
        df = pd.read_csv(input_path, sep=sep)

        df = self.clean_column(df)
        df = self.remove_duplicates(df, id_column="Job ID")

        df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned file at: {output_path}")
        return df


def main():
    logger.info("Starting text cleaning process...")
    parser = argparse.ArgumentParser(description="Clean job or resume datasets")
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV')
    parser.add_argument('--output', type=str, required=True, help='Path to save cleaned CSV')
    parser.add_argument('--sep', type=str, default=',', help='CSV separator (default: ,)')
    parser.add_argument('--columns', nargs='+', default=['Job Description'], help='Columns to clean')
    parser.add_argument('--idcol', type=str, default='Job Id', help='Column name for removing duplicates')
    args = parser.parse_args()

    cleaner = TextCleaner(columns_to_clean=args.columns)
    cleaner.process_file(
        input_path=args.input,
        output_path=args.output,
        sep=args.sep
    )
    logger.info("Text cleaning process finished successfully.")


if __name__ == "__main__":
    main()
