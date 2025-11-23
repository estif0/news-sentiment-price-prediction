import unittest
import pandas as pd
import os
import tempfile
from src.core.data_loader import DataLoader


class TestDataLoaderStock(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader()

        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()

        # Create test stock data
        self.test_stock_data = pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "Open": [100.0, 102.0, 101.5],
                "High": [105.0, 106.0, 103.0],
                "Low": [99.0, 101.0, 100.0],
                "Close": [103.0, 105.0, 102.0],
                "Volume": [1000000, 1200000, 800000],
            }
        )

        # Create valid test file
        self.test_file = os.path.join(self.temp_dir, "TEST.csv")
        self.test_stock_data.to_csv(self.test_file, index=False)

        # Create invalid test file (missing columns)
        self.invalid_file = os.path.join(self.temp_dir, "INVALID.csv")
        invalid_data = pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02"],
                "Price": [100.0, 102.0],  # Missing OHLCV columns
            }
        )
        invalid_data.to_csv(self.invalid_file, index=False)

    def tearDown(self):
        # Clean up temporary files
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_load_stock_data_success(self):
        """Test successful loading of valid stock data."""
        df = self.loader.load_stock_data(self.test_file)

        # Check DataFrame properties
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)

        # Check required columns exist (Date should be index)
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_columns:
            self.assertIn(col, df.columns)

        # Check that Date is the index
        self.assertEqual(df.index.name, "Date")
        self.assertTrue(isinstance(df.index, pd.DatetimeIndex))

        # Check data is sorted by date (index should be sorted)
        self.assertTrue(df.index.is_monotonic_increasing)

    def test_load_stock_data_file_not_found(self):
        """Test error handling when stock file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_stock_data("non_existent_file.csv")

    def test_load_stock_data_missing_columns(self):
        """Test error handling when required columns are missing."""
        with self.assertRaises(Exception) as context:
            self.loader.load_stock_data(self.invalid_file)

        self.assertIn("Missing required columns", str(context.exception))

    def test_load_multiple_stocks_success(self):
        """Test loading multiple stock files."""
        # Create another test file
        test_file2 = os.path.join(self.temp_dir, "TEST2.csv")
        self.test_stock_data.to_csv(test_file2, index=False)

        stock_data = self.loader.load_multiple_stocks(self.temp_dir)

        # Should load valid files and skip invalid ones
        self.assertIsInstance(stock_data, dict)
        self.assertIn("TEST", stock_data)
        self.assertIn("TEST2", stock_data)
        self.assertEqual(len(stock_data["TEST"]), 3)

    def test_load_multiple_stocks_specific_symbols(self):
        """Test loading specific stock symbols."""
        # Create another test file
        test_file2 = os.path.join(self.temp_dir, "AAPL.csv")
        self.test_stock_data.to_csv(test_file2, index=False)

        stock_data = self.loader.load_multiple_stocks(
            self.temp_dir, stock_symbols=["AAPL"]
        )

        self.assertEqual(len(stock_data), 1)
        self.assertIn("AAPL", stock_data)

    def test_load_multiple_stocks_directory_not_found(self):
        """Test error handling when directory doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_multiple_stocks("non_existent_directory")

    def test_validate_ohlcv_data(self):
        """Test OHLCV validation logic."""
        # Test with valid data
        try:
            self.loader._validate_ohlcv_data(self.test_stock_data)
        except Exception:
            self.fail("Validation failed on valid OHLCV data")

        # Test with invalid data (High < Low)
        invalid_data = self.test_stock_data.copy()
        invalid_data.loc[0, "High"] = 50.0  # Set High < Low

        with self.assertRaises(ValueError) as context:
            self.loader._validate_ohlcv_data(invalid_data)

        self.assertIn("High < Low", str(context.exception))


if __name__ == "__main__":
    unittest.main()
