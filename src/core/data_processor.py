import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class DataProcessor:
    """
    Class for processing and aligning news and stock data for correlation analysis.
    Handles date normalization, timezone conversion, and dataset merging.
    """

    def __init__(self):
        """Initialize the DataProcessor."""
        pass

    def normalize_news_dates(
        self, news_df: pd.DataFrame, date_column: str = "date"
    ) -> pd.DataFrame:
        """
        Normalize news dates from UTC-4 to trading dates.
        News published after market close (4:00 PM EST) or on weekends/holidays
        are mapped to the next trading day.

        Args:
            news_df (pd.DataFrame): News dataset with date column
            date_column (str): Name of the date column (default: 'date')

        Returns:
            pd.DataFrame: News dataset with normalized trading dates

        Raises:
            ValueError: If date column is missing or invalid
        """
        if date_column not in news_df.columns:
            raise ValueError(f"Date column '{date_column}' not found in news data")

        # Create a copy to avoid modifying original data
        df = news_df.copy()

        # Parse dates to datetime (already in UTC-4)
        df[date_column] = pd.to_datetime(df[date_column])

        # Extract date (without time) for alignment
        df["trading_date"] = df[date_column].dt.date

        # Convert to datetime for easier manipulation
        df["trading_date"] = pd.to_datetime(df["trading_date"])

        # Handle after-hours news (after 4:00 PM EST)
        # News published after market close should be considered for next trading day
        market_close_hour = 16  # 4:00 PM

        # Create mask for after-hours news
        after_hours_mask = df[date_column].dt.hour >= market_close_hour

        # Shift after-hours news to next day
        df.loc[after_hours_mask, "trading_date"] = df.loc[
            after_hours_mask, "trading_date"
        ] + pd.Timedelta(days=1)

        # Handle weekends: Saturday -> Monday, Sunday -> Monday
        # Monday = 0, Sunday = 6
        weekend_mask = df["trading_date"].dt.dayofweek >= 5

        # Calculate days to add to get to Monday
        days_to_monday = 7 - df["trading_date"].dt.dayofweek
        df.loc[weekend_mask, "trading_date"] = df.loc[
            weekend_mask, "trading_date"
        ] + pd.to_timedelta(days_to_monday[weekend_mask], unit="D")

        return df

    def align_with_trading_days(
        self,
        news_df: pd.DataFrame,
        stock_df: pd.DataFrame,
        stock_date_column: str = "Date",
    ) -> pd.DataFrame:
        """
        Align news dates with actual trading days from stock data.
        This ensures news is only associated with days when market was open.

        Args:
            news_df (pd.DataFrame): News dataset with trading_date column
            stock_df (pd.DataFrame): Stock dataset with date index or column
            stock_date_column (str): Name of date column in stock data (default: 'Date')

        Returns:
            pd.DataFrame: News dataset with dates aligned to actual trading days

        Raises:
            ValueError: If required columns are missing
        """
        if "trading_date" not in news_df.columns:
            raise ValueError("News data must have 'trading_date' column")

        # Create a copy
        df = news_df.copy()

        # Get stock trading dates
        if isinstance(stock_df.index, pd.DatetimeIndex):
            trading_dates = stock_df.index
        elif stock_date_column in stock_df.columns:
            trading_dates = pd.to_datetime(stock_df[stock_date_column])
        else:
            raise ValueError(
                f"Stock data must have DatetimeIndex or '{stock_date_column}' column"
            )

        # Convert to sorted list of unique dates
        trading_dates = (
            pd.Series(trading_dates.unique()).sort_values().reset_index(drop=True)
        )

        # For each news date, find the next available trading day
        def map_to_trading_day(date):
            """Map a date to the nearest trading day (forward-looking)."""
            # Find trading days >= news date
            future_days = trading_dates[trading_dates >= date]

            if len(future_days) > 0:
                return future_days.iloc[0]
            else:
                # If no future trading day, use last available
                return trading_dates.iloc[-1]

        df["aligned_date"] = df["trading_date"].apply(map_to_trading_day)

        return df

    def merge_news_stock_data(
        self,
        news_df: pd.DataFrame,
        stock_data: Dict[str, pd.DataFrame],
        stock_column: str = "stock",
    ) -> pd.DataFrame:
        """
        Merge news and stock data on aligned date and stock symbol.

        Args:
            news_df (pd.DataFrame): News dataset with aligned_date and stock columns
            stock_data (Dict[str, pd.DataFrame]): Dictionary of stock DataFrames by symbol
            stock_column (str): Name of stock symbol column in news data (default: 'stock')

        Returns:
            pd.DataFrame: Merged dataset with news and stock information

        Raises:
            ValueError: If required columns are missing
        """
        if "aligned_date" not in news_df.columns:
            raise ValueError("News data must have 'aligned_date' column")

        if stock_column not in news_df.columns:
            raise ValueError(f"Stock column '{stock_column}' not found in news data")

        # Create a copy
        merged_data = []

        # Group news by stock symbol
        for symbol, symbol_news in news_df.groupby(stock_column):
            if symbol not in stock_data:
                print(f"⚠️ Stock data not available for symbol: {symbol}")
                continue

            stock_df = stock_data[symbol].copy()

            # Reset index if needed to make Date a column
            if isinstance(stock_df.index, pd.DatetimeIndex):
                stock_df = stock_df.reset_index()
                stock_df.rename(columns={"index": "Date"}, inplace=True)

            # Ensure Date column exists
            if "Date" not in stock_df.columns:
                print(f"⚠️ Date column not found in stock data for {symbol}")
                continue

            # Convert Date to datetime without time component for merging
            stock_df["Date"] = pd.to_datetime(stock_df["Date"]).dt.normalize()

            # Prepare news data for merging
            news_for_merge = symbol_news.copy()
            news_for_merge["aligned_date"] = pd.to_datetime(
                news_for_merge["aligned_date"]
            ).dt.normalize()

            # Merge on aligned date
            merged = news_for_merge.merge(
                stock_df,
                left_on="aligned_date",
                right_on="Date",
                how="inner",
                suffixes=("_news", "_stock"),
            )

            merged["symbol"] = symbol
            merged_data.append(merged)

        if not merged_data:
            raise ValueError("No data could be merged. Check stock symbols and dates.")

        # Concatenate all merged data
        final_df = pd.concat(merged_data, ignore_index=True)

        return final_df

    def process_and_merge(
        self,
        news_df: pd.DataFrame,
        stock_data: Dict[str, pd.DataFrame],
        news_date_column: str = "date",
        stock_column: str = "stock",
    ) -> pd.DataFrame:
        """
        Complete pipeline to process and merge news with stock data.

        This method:
        1. Normalizes news dates from UTC-4 to trading dates
        2. Aligns dates with actual trading days
        3. Merges news and stock data by date and symbol

        Args:
            news_df (pd.DataFrame): Raw news dataset
            stock_data (Dict[str, pd.DataFrame]): Dictionary of stock DataFrames by symbol
            news_date_column (str): Name of date column in news data (default: 'date')
            stock_column (str): Name of stock symbol column in news data (default: 'stock')

        Returns:
            pd.DataFrame: Fully processed and merged dataset

        Example:
            >>> processor = DataProcessor()
            >>> merged_df = processor.process_and_merge(news_df, stock_dict)
        """
        print("Step 1: Normalizing news dates...")
        news_normalized = self.normalize_news_dates(news_df, news_date_column)

        print("Step 2: Aligning with trading days...")
        # Use first available stock for trading day reference
        reference_stock = list(stock_data.values())[0]
        news_aligned = self.align_with_trading_days(news_normalized, reference_stock)

        print("Step 3: Merging news and stock data...")
        merged_df = self.merge_news_stock_data(news_aligned, stock_data, stock_column)

        print(f"✅ Merge complete: {len(merged_df)} records")

        return merged_df

    def validate_merge_quality(self, merged_df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate the quality of the merged dataset.

        Args:
            merged_df (pd.DataFrame): Merged news and stock dataset

        Returns:
            Dict[str, any]: Dictionary containing validation metrics
        """
        metrics = {
            "total_records": len(merged_df),
            "unique_dates": merged_df["aligned_date"].nunique(),
            "unique_stocks": merged_df["symbol"].nunique(),
            "date_range": (
                merged_df["aligned_date"].min(),
                merged_df["aligned_date"].max(),
            ),
            "missing_values": merged_df.isnull().sum().to_dict(),
            "records_per_stock": merged_df["symbol"].value_counts().to_dict(),
        }

        return metrics

    def aggregate_daily_news(
        self, merged_df: pd.DataFrame, aggregation_columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Aggregate multiple news articles per stock per day.
        Useful when multiple news items exist for the same stock on the same day.

        Args:
            merged_df (pd.DataFrame): Merged dataset
            aggregation_columns (List[str]): Columns to aggregate (default: None, returns count)

        Returns:
            pd.DataFrame: Aggregated dataset with one row per stock per day
        """
        if aggregation_columns is None:
            # Just count articles per day per stock
            agg_df = (
                merged_df.groupby(["symbol", "aligned_date"])
                .size()
                .reset_index(name="article_count")
            )
        else:
            # Aggregate specified columns
            agg_dict = {col: "mean" for col in aggregation_columns}
            agg_df = (
                merged_df.groupby(["symbol", "aligned_date"])
                .agg(agg_dict)
                .reset_index()
            )

        return agg_df
