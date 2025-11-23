import pandas as pd
from typing import List, Tuple
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os

# Define local NLTK data path in the project root
# src/core/eda.py -> ../../ -> project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
nltk_data_path = os.path.join(project_root, 'nltk_data')

if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Add to NLTK path
if nltk_data_path not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_path)

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except (LookupError, OSError):
    print(f"Downloading NLTK data to {nltk_data_path}...")
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('punkt_tab', download_dir=nltk_data_path)
    nltk.download('stopwords', download_dir=nltk_data_path)
except Exception as e:
    # Fallback for corrupted zip files or other issues
    print(f"Error finding NLTK data: {e}, forcing download to {nltk_data_path}...")
    nltk.download('punkt', download_dir=nltk_data_path, force=True)
    nltk.download('punkt_tab', download_dir=nltk_data_path, force=True)
    nltk.download('stopwords', download_dir=nltk_data_path, force=True)

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
        self._headline_lengths = None
        self._publisher_counts = None
        self._common_keywords = {} # Cache by top_n
        self._parsed_dates = None
        self._publication_frequency = {} # Cache by freq
        self._publishing_times = None
        self._publisher_domains = None
        self._publisher_stock_counts = None

    def calculate_headline_lengths(self) -> pd.Series:
        """
        Calculates the length of each headline.

        Returns:
            pd.Series: A series containing the length of each headline.
        """
        if self._headline_lengths is None:
            if 'headline' not in self.df.columns:
                raise ValueError("DataFrame must contain a 'headline' column.")
            self._headline_lengths = self.df['headline'].astype(str).apply(len)
        return self._headline_lengths

    def count_articles_per_publisher(self) -> pd.Series:
        """
        Counts the number of articles per publisher.

        Returns:
            pd.Series: A series with publisher names as index and counts as values.
        """
        if self._publisher_counts is None:
            if 'publisher' not in self.df.columns:
                raise ValueError("DataFrame must contain a 'publisher' column.")
            self._publisher_counts = self.df['publisher'].value_counts()
        return self._publisher_counts

    def extract_common_keywords(self, column: str = 'headline', top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Extracts common keywords from a text column using NLTK for tokenization and stopword removal.
        
        Args:
            column (str): The column to analyze.
            top_n (int): The number of top keywords to return.

        Returns:
            List[Tuple[str, int]]: A list of tuples containing (keyword, count).
        """
        if top_n in self._common_keywords:
            return self._common_keywords[top_n]

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
        
        result = Counter(filtered_tokens).most_common(top_n)
        self._common_keywords[top_n] = result
        return result

    def parse_dates(self, date_column: str = 'date') -> pd.Series:
        """
        Parses the date column into datetime objects.

        Args:
            date_column (str): The name of the date column.

        Returns:
            pd.Series: The parsed datetime column.
        """
        if self._parsed_dates is None:
            if date_column not in self.df.columns:
                raise ValueError(f"DataFrame must contain a '{date_column}' column.")
            
            # Assuming the date format is consistent, but using coerce to handle errors
            # The dataset description says UTC-4, but pandas usually handles this well if format is standard
            self._parsed_dates = pd.to_datetime(self.df[date_column], errors='coerce', utc=True)
            
        return self._parsed_dates

    def analyze_publication_frequency(self, date_column: str = 'date', freq: str = 'D') -> pd.Series:
        """
        Analyzes the frequency of articles over time.

        Args:
            date_column (str): The name of the date column.
            freq (str): The frequency string (e.g., 'D' for daily, 'M' for monthly).

        Returns:
            pd.Series: A series with the date as index and article count as values.
        """
        if freq in self._publication_frequency:
            return self._publication_frequency[freq]

        if date_column not in self.df.columns:
            raise ValueError(f"DataFrame must contain a '{date_column}' column.")
            
        # Ensure dates are datetime objects
        if not pd.api.types.is_datetime64_any_dtype(self.df[date_column]):
             self.df[date_column] = self.parse_dates(date_column)

        result = self.df.set_index(date_column).resample(freq).size()
        self._publication_frequency[freq] = result
        return result

    def analyze_publishing_times(self, date_column: str = 'date') -> pd.Series:
        """
        Analyzes the time of day when articles are published.

        Args:
            date_column (str): The name of the date column.

        Returns:
            pd.Series: A series with the hour of day as index and article count as values.
        """
        if self._publishing_times is None:
            if date_column not in self.df.columns:
                raise ValueError(f"DataFrame must contain a '{date_column}' column.")

            # Ensure dates are datetime objects
            if not pd.api.types.is_datetime64_any_dtype(self.df[date_column]):
                 self.df[date_column] = self.parse_dates(date_column)
                 
            self._publishing_times = self.df[date_column].dt.hour.value_counts().sort_index()
        return self._publishing_times

    def extract_publisher_domains(self, publisher_column: str = 'publisher') -> pd.Series:
        """
        Extracts domains from publisher email addresses.
        
        Args:
            publisher_column (str): The name of the publisher column.

        Returns:
            pd.Series: A series containing the extracted domains (or None if not an email).
        """
        if self._publisher_domains is None:
            if publisher_column not in self.df.columns:
                raise ValueError(f"DataFrame must contain a '{publisher_column}' column.")

            self._publisher_domains = self.df[publisher_column].astype(str).apply(lambda x: x.split('@')[-1] if '@' in x else None)
        return self._publisher_domains

    def analyze_publisher_stock_counts(self, publisher_column: str = 'publisher', stock_column: str = 'stock') -> pd.DataFrame:
        """
        Counts how many articles each publisher has written for each stock.
        
        Args:
            publisher_column (str): The name of the publisher column.
            stock_column (str): The name of the stock column.

        Returns:
            pd.DataFrame: DataFrame with columns [publisher, stock, count], sorted by count descending.
        """
        if self._publisher_stock_counts is None:
            if publisher_column not in self.df.columns or stock_column not in self.df.columns:
                raise ValueError(f"DataFrame must contain '{publisher_column}' and '{stock_column}' columns.")
                
            self._publisher_stock_counts = self.df.groupby([publisher_column, stock_column]).size().reset_index(name='count').sort_values('count', ascending=False)
        return self._publisher_stock_counts
