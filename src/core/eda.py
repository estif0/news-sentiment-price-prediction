import pandas as pd
from typing import List, Tuple
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

class EDAAnalyzer:
    """
    Class for performing Exploratory Data Analysis on financial news data.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): The dataframe containing news data.
        """
        self.df = df

    def calculate_headline_lengths(self) -> pd.Series:
        """
        Calculates the length of each headline.

        Returns:
            pd.Series: A series containing the length of each headline.
        """
        if 'headline' not in self.df.columns:
            raise ValueError("DataFrame must contain a 'headline' column.")
        return self.df['headline'].astype(str).apply(len)

    def count_articles_per_publisher(self) -> pd.Series:
        """
        Counts the number of articles per publisher.

        Returns:
            pd.Series: A series with publisher names as index and counts as values.
        """
        if 'publisher' not in self.df.columns:
            raise ValueError("DataFrame must contain a 'publisher' column.")
        return self.df['publisher'].value_counts()

    def extract_common_keywords(self, column: str = 'headline', top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Extracts common keywords from a text column using NLTK for tokenization and stopword removal.
        
        Args:
            column (str): The column to analyze.
            top_n (int): The number of top keywords to return.

        Returns:
            List[Tuple[str, int]]: A list of tuples containing (keyword, count).
        """
        if column not in self.df.columns:
            raise ValueError(f"DataFrame must contain a '{column}' column.")
            
        # Combine all text
        text = ' '.join(self.df[column].astype(str).tolist()).lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove punctuation and stopwords
        stop_words = set(stopwords.words('english'))
        punctuation = set(string.punctuation)
        
        filtered_tokens = [
            word for word in tokens 
            if word not in stop_words and word not in punctuation and len(word) > 2
        ]
        
        return Counter(filtered_tokens).most_common(top_n)
