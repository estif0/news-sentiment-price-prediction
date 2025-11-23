import pandas as pd
import os
from typing import Optional


class DataLoader:
    """
    A class to load financial news and stock price data.
    """

    def load_news_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads the financial news dataset from a CSV file.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: If there is an error reading the file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")

        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
