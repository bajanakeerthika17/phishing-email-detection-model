"""
preprocessing.py
----------------
Module for cleaning and preprocessing email text and datasets for Phishing Email Detection.

Key Functions:
1. clean_dataset: Cleans missing values and duplicate records.
2. clean_text: Normalizes text (lowercase, removes special characters, tokenization, stop words).
3. tokenize_text: Splits text into clean word tokens.
4. get_tfidf_vectorizer: Creates and configures a TF-IDF vectorizer.
"""

import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Standard list of common English stopwords for lightweight preprocessing
# (Avoids heavy NLTK download requirements for beginner ease of use)
STANDARD_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
    "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
    "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves"
}


def clean_dataset(df: pd.DataFrame, text_col: str = "email_text", label_col: str = "label") -> pd.DataFrame:
    """
    Cleans a pandas DataFrame:
    1. Removes rows with missing (null) values in critical columns.
    2. Removes duplicate records based on email text.
    3. Resets the index and returns the cleaned DataFrame.

    Parameters:
        df (pd.DataFrame): The raw input dataframe.
        text_col (str): Column name containing the email text.
        label_col (str): Column name containing the classification label.

    Returns:
        pd.DataFrame: Cleaned DataFrame without nulls or duplicates.
    """
    initial_count = len(df)
    
    # 1. Drop missing values in text and label columns
    df_clean = df.dropna(subset=[text_col, label_col]).copy()
    null_dropped = initial_count - len(df_clean)

    # 2. Drop duplicate email records
    df_clean = df_clean.drop_duplicates(subset=[text_col]).copy()
    duplicates_dropped = initial_count - null_dropped - len(df_clean)

    # 3. Ensure label column is clean integer (0 or 1)
    df_clean[label_col] = df_clean[label_col].astype(int)

    # Reset DataFrame index
    df_clean = df_clean.reset_index(drop=True)

    print(f"[Preprocessing] Initial records: {initial_count}")
    print(f"[Preprocessing] Removed nulls: {null_dropped}")
    print(f"[Preprocessing] Removed duplicates: {duplicates_dropped}")
    print(f"[Preprocessing] Cleaned dataset records: {len(df_clean)}")

    return df_clean


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Preprocesses raw email text:
    1. Converts text to lowercase.
    2. Strips HTML tags if present.
    3. Normalizes whitespace and removes redundant special characters.
    4. Performs tokenization and optional stopword removal.

    Parameters:
        text (str): Raw email text string.
        remove_stopwords (bool): Whether to filter out common English stopwords.

    Returns:
        str: Cleaned, normalized string.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # Convert to lowercase
    text = text.lower().strip()

    # Strip HTML tags like <br>, <p>, etc.
    text = re.sub(r"<[^>]+>", " ", text)

    # Replace newlines, tabs, and carriage returns with single space
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Remove URLs temporarily from text normalization so tokens stay clean
    # (URLs are separately analyzed in feature_extraction.py)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove punctuation & special characters (keep alphanumeric and spaces)
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra spaces
    tokens = text.split()

    # Stopword removal where appropriate
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STANDARD_STOPWORDS and len(t) > 1]

    return " ".join(tokens)


def tokenize_text(text: str) -> list:
    """
    Tokenizes email text into individual word tokens.

    Parameters:
        text (str): Input text.

    Returns:
        list: List of word tokens.
    """
    cleaned = clean_text(text, remove_stopwords=False)
    return cleaned.split()


def get_tfidf_vectorizer(max_features: int = 3000, ngram_range: tuple = (1, 2)) -> TfidfVectorizer:
    """
    Creates and returns a configured Scikit-learn TF-IDF Vectorizer.
    
    TF-IDF (Term Frequency - Inverse Document Frequency) translates text into
    numerical matrices, weighting unique terms heavily while reducing common ones.

    Parameters:
        max_features (int): Maximum vocabulary size to prevent overfitting.
        ngram_range (tuple): Range of n-grams (1, 2) includes both unigrams and bigrams.

    Returns:
        TfidfVectorizer: Configured vectorizer instance.
    """
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True
    )
