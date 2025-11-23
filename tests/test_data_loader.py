import unittest
import pandas as pd
import os
from src.core.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader()
        self.test_file = "test_news_data.csv"
        # Create a dummy CSV file for testing
        df = pd.DataFrame(
            {
                "headline": ["News 1", "News 2"],
                "url": ["http://url1", "http://url2"],
                "publisher": ["Pub 1", "Pub 2"],
                "date": ["2023-01-01", "2023-01-02"],
                "stock": ["AAPL", "GOOG"],
            }
        )
        df.to_csv(self.test_file, index=False)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_load_news_data(self):
        df = self.loader.load_news_data(self.test_file)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertListEqual(
            list(df.columns), ["headline", "url", "publisher", "date", "stock"]
        )

    def test_load_news_data_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.loader.load_news_data("non_existent_file.csv")


if __name__ == "__main__":
    unittest.main()
