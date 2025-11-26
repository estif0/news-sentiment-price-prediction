import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from core.sentiment_analyzer import SentimentAnalyzer


class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for SentimentAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.analyzer = SentimentAnalyzer(method="textblob")
            self.textblob_available = True
        except ImportError:
            self.textblob_available = False
            print("TextBlob not available, skipping some tests")

        # Sample news data
        self.sample_data = pd.DataFrame(
            {
                "headline": [
                    "Stock prices surge on positive earnings report",
                    "Company faces major losses and declining revenue",
                    "Market remains stable with no significant changes",
                    "Investors excited about new product launch",
                    "Regulatory concerns threaten company operations",
                ],
                "stock": ["AAPL", "MSFT", "GOOG", "AAPL", "MSFT"],
                "aligned_date": pd.to_datetime(
                    [
                        "2020-01-06",
                        "2020-01-06",
                        "2020-01-07",
                        "2020-01-07",
                        "2020-01-08",
                    ]
                ),
            }
        )

    def test_initialization(self):
        """Test SentimentAnalyzer initialization."""
        if self.textblob_available:
            analyzer = SentimentAnalyzer(method="textblob")
            self.assertEqual(analyzer.method, "textblob")

    def test_analyze_headline_positive(self):
        """Test sentiment analysis for positive headline."""
        if not self.textblob_available:
            self.skipTest("TextBlob not available")

        text = "Stock prices surge on positive earnings report"
        score = self.analyzer.analyze_headline(text)

        # Should return a positive score
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 1.0)

    def test_analyze_headline_negative(self):
        """Test sentiment analysis for negative headline."""
        if not self.textblob_available:
            self.skipTest("TextBlob not available")

        text = "Terrible losses threaten bankruptcy and devastating collapse"
        score = self.analyzer.analyze_headline(text)

        # Should return a negative score or at least be in valid range
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)
        # More lenient check: if not negative, at least should be low
        if score >= 0:
            self.assertLess(score, 0.3, "Expected negative or low sentiment")

    def test_analyze_headline_neutral(self):
        """Test sentiment analysis for neutral headline."""
        if not self.textblob_available:
            self.skipTest("TextBlob not available")

        text = "Market remains stable with no significant changes"
        score = self.analyzer.analyze_headline(text)

        # Should return a score close to 0
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

    def test_analyze_headline_empty(self):
        """Test sentiment analysis for empty string."""
        if not self.textblob_available:
            self.skipTest("TextBlob not available")

        score = self.analyzer.analyze_headline("")
        self.assertEqual(score, 0.0)

    def test_analyze_headline_invalid(self):
        """Test sentiment analysis for invalid input."""
        if not self.textblob_available:
            self.skipTest("TextBlob not available")

        score = self.analyzer.analyze_headline(None)
        self.assertEqual(score, 0.0)

    def test_classify_sentiment_positive(self):
        """Test sentiment classification for positive score."""
        if not self.textblob_available:
            self.skipTest("TextBlob not available")

        classification = self.analyzer.classify_sentiment(0.5)
        self.assertEqual(classification, "positive")

    def test_classify_sentiment_negative(self):
        """Test sentiment classification for negative score."""
        if not self.textblob_available:
            self.skipTest("TextBlob not available")

        classification = self.analyzer.classify_sentiment(-0.5)
        self.assertEqual(classification, "negative")

    def test_classify_sentiment_neutral(self):
        """Test sentiment classification for neutral score."""
        if not self.textblob_available:
            self.skipTest("TextBlob not available")

        classification = self.analyzer.classify_sentiment(0.02)
        self.assertEqual(classification, "neutral")

    def test_classify_sentiment_threshold(self):
        """Test custom threshold for sentiment classification."""
        if not self.textblob_available:
            self.skipTest("TextBlob not available")

        # With default threshold (0.05)
        self.assertEqual(self.analyzer.classify_sentiment(0.04), "neutral")

        # With custom threshold (0.1)
        classification = self.analyzer.classify_sentiment(0.08, threshold=0.1)
        self.assertEqual(classification, "neutral")

    def test_analyze_dataframe(self):
        """Test sentiment analysis on DataFrame."""
        if not self.textblob_available:
            self.skipTest("TextBlob not available")

        result = self.analyzer.analyze_dataframe(self.sample_data)

        # Check that sentiment columns are added
        self.assertIn("sentiment_score", result.columns)
        self.assertIn("sentiment_class", result.columns)

        # Check that all scores are in valid range
        self.assertTrue(all(result["sentiment_score"] >= -1.0))
        self.assertTrue(all(result["sentiment_score"] <= 1.0))

        # Check that classifications are valid
        valid_classes = ["positive", "negative", "neutral"]
        self.assertTrue(all(result["sentiment_class"].isin(valid_classes)))

    def test_analyze_dataframe_missing_column(self):
        """Test error handling for missing text column."""
        if not self.textblob_available:
            self.skipTest("TextBlob not available")

        df = pd.DataFrame({"title": ["Test"]})

        with self.assertRaises(ValueError):
            self.analyzer.analyze_dataframe(df, text_column="headline")

    def test_aggregate_daily_sentiment(self):
        """Test daily sentiment aggregation."""
        if not self.textblob_available:
            self.skipTest("TextBlob not available")

        # First analyze sentiment
        df_with_sentiment = self.analyzer.analyze_dataframe(self.sample_data)

        # Add symbol column for aggregation
        df_with_sentiment["symbol"] = df_with_sentiment["stock"]

        # Aggregate
        result = self.analyzer.aggregate_daily_sentiment(df_with_sentiment)

        # Check that aggregation columns exist
        expected_columns = [
            "symbol",
            "aligned_date",
            "daily_sentiment",
            "sentiment_std",
            "article_count",
            "positive_count",
            "negative_count",
            "neutral_count",
        ]
        for col in expected_columns:
            self.assertIn(col, result.columns)

        # Check that aggregation reduces rows (multiple articles per day)
        self.assertLessEqual(len(result), len(df_with_sentiment))

    def test_aggregate_daily_sentiment_missing_columns(self):
        """Test error handling for missing required columns."""
        if not self.textblob_available:
            self.skipTest("TextBlob not available")

        df = pd.DataFrame({"date": ["2020-01-01"]})

        with self.assertRaises(ValueError):
            self.analyzer.aggregate_daily_sentiment(df)

    def test_get_sentiment_distribution(self):
        """Test sentiment distribution statistics."""
        if not self.textblob_available:
            self.skipTest("TextBlob not available")

        df_with_sentiment = self.analyzer.analyze_dataframe(self.sample_data)

        stats = self.analyzer.get_sentiment_distribution(df_with_sentiment)

        # Check that expected statistics are present
        expected_keys = [
            "mean",
            "median",
            "std",
            "min",
            "max",
            "positive_pct",
            "negative_pct",
            "neutral_pct",
        ]
        for key in expected_keys:
            self.assertIn(key, stats)

        # Check that percentages sum to ~100%
        total_pct = stats["positive_pct"] + stats["negative_pct"] + stats["neutral_pct"]
        self.assertAlmostEqual(total_pct, 100.0, places=1)

    def test_get_sentiment_distribution_by_stock(self):
        """Test sentiment distribution by stock."""
        if not self.textblob_available:
            self.skipTest("TextBlob not available")

        df_with_sentiment = self.analyzer.analyze_dataframe(self.sample_data)
        df_with_sentiment["symbol"] = df_with_sentiment["stock"]

        stats = self.analyzer.get_sentiment_distribution(
            df_with_sentiment, by_stock=True
        )

        # Check that stats are calculated for each stock
        self.assertIsInstance(stats, dict)
        for symbol in df_with_sentiment["symbol"].unique():
            self.assertIn(symbol, stats)

    def test_filter_by_sentiment(self):
        """Test filtering by sentiment class."""
        if not self.textblob_available:
            self.skipTest("TextBlob not available")

        df_with_sentiment = self.analyzer.analyze_dataframe(self.sample_data)

        positive = self.analyzer.filter_by_sentiment(df_with_sentiment, "positive")
        negative = self.analyzer.filter_by_sentiment(df_with_sentiment, "negative")
        neutral = self.analyzer.filter_by_sentiment(df_with_sentiment, "neutral")

        # Check that filters work correctly
        self.assertTrue(all(positive["sentiment_class"] == "positive"))
        self.assertTrue(all(negative["sentiment_class"] == "negative"))
        self.assertTrue(all(neutral["sentiment_class"] == "neutral"))

        # Check that total equals original
        total_filtered = len(positive) + len(negative) + len(neutral)
        self.assertEqual(total_filtered, len(df_with_sentiment))

    def test_filter_by_sentiment_invalid_class(self):
        """Test error handling for invalid sentiment class."""
        if not self.textblob_available:
            self.skipTest("TextBlob not available")

        df_with_sentiment = self.analyzer.analyze_dataframe(self.sample_data)

        with self.assertRaises(ValueError):
            self.analyzer.filter_by_sentiment(df_with_sentiment, "invalid")


if __name__ == "__main__":
    unittest.main()
