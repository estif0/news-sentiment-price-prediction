import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from core.data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = DataProcessor()

        # Create sample news data
        dates = [
            "2020-01-06 10:30:00-04:00",  # Monday morning
            "2020-01-06 16:30:00-04:00",  # Monday after hours
            "2020-01-10 14:00:00-04:00",  # Friday afternoon
            "2020-01-11 10:00:00-04:00",  # Saturday (should map to Monday)
            "2020-01-12 10:00:00-04:00",  # Sunday (should map to Monday)
        ]

        self.sample_news = pd.DataFrame(
            {
                "date": dates,
                "headline": ["News 1", "News 2", "News 3", "News 4", "News 5"],
                "stock": ["AAPL", "AAPL", "MSFT", "AAPL", "MSFT"],
            }
        )

        # Create sample stock data for AAPL
        stock_dates = pd.date_range(start="2020-01-06", end="2020-01-17", freq="B")
        self.sample_stock = pd.DataFrame(
            {
                "Date": stock_dates,
                "Open": np.random.uniform(100, 110, len(stock_dates)),
                "High": np.random.uniform(110, 120, len(stock_dates)),
                "Low": np.random.uniform(90, 100, len(stock_dates)),
                "Close": np.random.uniform(100, 110, len(stock_dates)),
                "Volume": np.random.randint(1000000, 5000000, len(stock_dates)),
            }
        )

    def test_normalize_news_dates(self):
        """Test normalization of news dates."""
        result = self.processor.normalize_news_dates(self.sample_news)

        # Check that trading_date column is created
        self.assertIn("trading_date", result.columns)

        # Check that dates are normalized to date only (no time)
        self.assertTrue(
            all(
                pd.to_datetime(result["trading_date"]).dt.time
                == pd.Timestamp("00:00:00").time()
            )
        )

    def test_normalize_dates_after_hours(self):
        """Test that after-hours news is mapped to next day."""
        result = self.processor.normalize_news_dates(self.sample_news)

        # Second news item is after hours (16:30), should be next day
        original_date = pd.to_datetime(self.sample_news.iloc[1]["date"]).date()
        trading_date = result.iloc[1]["trading_date"].date()

        # Should be one day later
        self.assertEqual(trading_date, original_date + timedelta(days=1))

    def test_normalize_dates_weekend(self):
        """Test that weekend dates are mapped to Monday."""
        result = self.processor.normalize_news_dates(self.sample_news)

        # Fourth item is Saturday, should map to Monday
        saturday_trading_date = result.iloc[3]["trading_date"]
        self.assertEqual(saturday_trading_date.dayofweek, 0)  # Monday

        # Fifth item is Sunday, should map to Monday
        sunday_trading_date = result.iloc[4]["trading_date"]
        self.assertEqual(sunday_trading_date.dayofweek, 0)  # Monday

    def test_normalize_dates_missing_column(self):
        """Test error handling for missing date column."""
        df = pd.DataFrame({"headline": ["Test"]})

        with self.assertRaises(ValueError):
            self.processor.normalize_news_dates(df)

    def test_align_with_trading_days(self):
        """Test alignment with actual trading days."""
        # First normalize dates
        news_normalized = self.processor.normalize_news_dates(self.sample_news)

        # Then align with trading days
        result = self.processor.align_with_trading_days(
            news_normalized, self.sample_stock
        )

        # Check that aligned_date column is created
        self.assertIn("aligned_date", result.columns)

        # Check that all aligned dates are in stock trading dates
        stock_dates = pd.to_datetime(self.sample_stock["Date"]).dt.normalize()
        for aligned_date in result["aligned_date"]:
            self.assertIn(pd.to_datetime(aligned_date).normalize(), stock_dates.values)

    def test_align_with_trading_days_missing_column(self):
        """Test error handling when trading_date column is missing."""
        df = pd.DataFrame({"headline": ["Test"]})

        with self.assertRaises(ValueError):
            self.processor.align_with_trading_days(df, self.sample_stock)

    def test_merge_news_stock_data(self):
        """Test merging of news and stock data."""
        # Prepare news data
        news_normalized = self.processor.normalize_news_dates(self.sample_news)
        news_aligned = self.processor.align_with_trading_days(
            news_normalized, self.sample_stock
        )

        # Create stock data dictionary
        stock_data = {"AAPL": self.sample_stock}

        # Merge
        result = self.processor.merge_news_stock_data(news_aligned, stock_data)

        # Check that result has both news and stock columns
        self.assertIn("headline", result.columns)
        self.assertIn("Close", result.columns)
        self.assertIn("symbol", result.columns)

        # Check that all symbols in result exist in stock_data
        for symbol in result["symbol"].unique():
            self.assertIn(symbol, stock_data.keys())

    def test_process_and_merge_pipeline(self):
        """Test complete pipeline from raw data to merged dataset."""
        stock_data = {"AAPL": self.sample_stock, "MSFT": self.sample_stock.copy()}

        result = self.processor.process_and_merge(self.sample_news, stock_data)

        # Check that result is not empty
        self.assertGreater(len(result), 0)

        # Check that required columns exist
        required_columns = ["headline", "stock", "Close", "symbol", "aligned_date"]
        for col in required_columns:
            self.assertIn(col, result.columns)

    def test_validate_merge_quality(self):
        """Test merge quality validation."""
        stock_data = {"AAPL": self.sample_stock}
        merged_df = self.processor.process_and_merge(self.sample_news, stock_data)

        metrics = self.processor.validate_merge_quality(merged_df)

        # Check that metrics dictionary has expected keys
        expected_keys = [
            "total_records",
            "unique_dates",
            "unique_stocks",
            "date_range",
            "missing_values",
            "records_per_stock",
        ]
        for key in expected_keys:
            self.assertIn(key, metrics)

        # Check that total_records matches DataFrame length
        self.assertEqual(metrics["total_records"], len(merged_df))

    def test_aggregate_daily_news(self):
        """Test daily news aggregation."""
        stock_data = {"AAPL": self.sample_stock}
        merged_df = self.processor.process_and_merge(self.sample_news, stock_data)

        result = self.processor.aggregate_daily_news(merged_df)

        # Check that result has aggregated columns
        self.assertIn("article_count", result.columns)
        self.assertIn("symbol", result.columns)
        self.assertIn("aligned_date", result.columns)

        # Check that each date-symbol combination appears only once
        self.assertEqual(
            len(result),
            merged_df.groupby(["symbol", "aligned_date"]).ngroups,
        )


if __name__ == "__main__":
    unittest.main()
