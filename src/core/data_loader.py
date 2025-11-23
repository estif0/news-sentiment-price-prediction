import pandas as pd
import os
from typing import List, Optional


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

    def load_stock_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads stock price data from a CSV file.

        Args:
            file_path (str): The path to the stock CSV file.

        Returns:
            pd.DataFrame: The loaded stock data as a pandas DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If required columns are missing or data validation fails.
            Exception: If there is an error reading the file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Stock data file not found at {file_path}")

        try:
            df = pd.read_csv(file_path)

            # Validate required columns
            required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Convert Date column to datetime and set as index
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")

            # Validate OHLCV relationships
            self._validate_ohlcv_data(df)

            # Sort by date to ensure chronological order
            df = df.sort_index()

            return df

        except Exception as e:
            raise Exception(f"Error loading stock data: {e}")

    def load_multiple_stocks(
        self,
        data_directory: str = None,
        stock_symbols: Optional[List[str]] = None,
        file_paths: List[str] = None,
    ) -> dict:
        """
        Loads multiple stock data files from directory or file paths.

        Args:
            data_directory (str, optional): Path to directory containing stock CSV files
            stock_symbols (Optional[List[str]]): List of stock symbols to load from directory
            file_paths (List[str], optional): List of specific file paths to load

        Returns:
            dict: Dictionary with stock symbols as keys and DataFrames as values.

        Raises:
            FileNotFoundError: If the directory does not exist.
            Exception: If there is an error loading any stock data.
        """
        stock_data = {}

        # Handle file_paths input (explicit file list)
        if file_paths:
            for file_path in file_paths:
                if os.path.exists(file_path):
                    symbol = os.path.basename(file_path).replace(".csv", "").upper()
                    try:
                        stock_data[symbol] = self.load_stock_data(file_path)
                        print(f"✅ Loaded {symbol}: {len(stock_data[symbol])} records")
                    except Exception as e:
                        print(f"❌ Failed to load {symbol}: {e}")
                        continue
                else:
                    print(f"⚠️ File not found: {file_path}")
            return stock_data

        # Handle directory input
        if not data_directory:
            raise ValueError("Either data_directory or file_paths must be provided")

        if not os.path.exists(data_directory):
            raise FileNotFoundError(f"Data directory not found: {data_directory}")

        stock_data = {}

        # Get list of files to process
        if stock_symbols:
            file_names = [f"{symbol}.csv" for symbol in stock_symbols]
        else:
            file_names = [f for f in os.listdir(data_directory) if f.endswith(".csv")]

        for file_name in file_names:
            file_path = os.path.join(data_directory, file_name)

            if os.path.exists(file_path):
                symbol = file_name.replace(".csv", "")
                try:
                    stock_data[symbol] = self.load_stock_data(file_path)
                    print(f"✅ Loaded {symbol}: {len(stock_data[symbol])} records")
                except Exception as e:
                    print(f"❌ Failed to load {symbol}: {e}")
                    continue
            else:
                print(f"⚠️ File not found: {file_path}")

        if not stock_data:
            raise Exception("No stock data files were successfully loaded")

        return stock_data

    def _validate_ohlcv_data(self, df: pd.DataFrame) -> None:
        """
        Validates OHLCV data relationships and data quality.

        Args:
            df (pd.DataFrame): Stock data DataFrame to validate.

        Raises:
            ValueError: If data validation fails.
        """
        # Check for negative values
        numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_columns:
            if (df[col] < 0).any():
                raise ValueError(f"Found negative values in {col} column")

        # Validate High >= Low relationship
        if (df["High"] < df["Low"]).any():
            raise ValueError("Found records where High < Low")

        # Validate High is the maximum of OHLC
        max_ohlc = df[["Open", "High", "Low", "Close"]].max(axis=1)
        if not (df["High"] >= max_ohlc).all():
            raise ValueError("High is not the maximum of Open, High, Low, Close")

        # Validate Low is the minimum of OHLC
        min_ohlc = df[["Open", "High", "Low", "Close"]].min(axis=1)
        if not (df["Low"] <= min_ohlc).all():
            raise ValueError("Low is not the minimum of Open, High, Low, Close")
