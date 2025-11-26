import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    from textblob import TextBlob

    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("Warning: TextBlob not available. Install it with: pip install textblob")

try:
    import nltk

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Install it with: pip install nltk")


class SentimentAnalyzer:
    """
    Class for performing sentiment analysis on financial news headlines.
    Uses TextBlob and NLTK for natural language processing.
    """

    def __init__(self, method: str = "textblob"):
        """
        Initialize the SentimentAnalyzer.

        Args:
            method (str): Sentiment analysis method ('textblob' or 'nltk')
        """
        self.method = method

        if method == "textblob" and not TEXTBLOB_AVAILABLE:
            raise ImportError(
                "TextBlob is not installed. Install it with: pip install textblob"
            )

        if method == "nltk" and not NLTK_AVAILABLE:
            raise ImportError(
                "NLTK is not installed. Install it with: pip install nltk"
            )

    def analyze_headline(self, text: str) -> float:
        """
        Analyze sentiment of a single headline.

        Args:
            text (str): News headline text

        Returns:
            float: Sentiment polarity score (-1 to +1)
                  -1 = very negative, 0 = neutral, +1 = very positive

        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> score = analyzer.analyze_headline("Stock prices surge on positive earnings")
            >>> print(score)  # Expected: positive value (e.g., 0.5)
        """
        if not isinstance(text, str) or not text.strip():
            return 0.0

        if self.method == "textblob":
            return self._analyze_textblob(text)
        elif self.method == "nltk":
            return self._analyze_nltk(text)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _analyze_textblob(self, text: str) -> float:
        """
        Analyze sentiment using TextBlob.

        Args:
            text (str): Text to analyze

        Returns:
            float: Polarity score (-1 to +1)
        """
        blob = TextBlob(text)
        return blob.sentiment.polarity

    def _analyze_nltk(self, text: str) -> float:
        """
        Analyze sentiment using NLTK's VADER.

        Args:
            text (str): Text to analyze

        Returns:
            float: Compound sentiment score (-1 to +1)
        """
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer

            sia = SentimentIntensityAnalyzer()
            scores = sia.polarity_scores(text)
            return scores["compound"]
        except LookupError:
            print("NLTK VADER lexicon not found. Downloading...")
            nltk.download("vader_lexicon")
            from nltk.sentiment import SentimentIntensityAnalyzer

            sia = SentimentIntensityAnalyzer()
            scores = sia.polarity_scores(text)
            return scores["compound"]

    def classify_sentiment(self, score: float, threshold: float = 0.05) -> str:
        """
        Classify sentiment score into categories.

        Args:
            score (float): Sentiment polarity score (-1 to +1)
            threshold (float): Threshold for neutral classification (default: 0.05)

        Returns:
            str: Sentiment category ('positive', 'negative', 'neutral')

        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> analyzer.classify_sentiment(0.5)
            'positive'
            >>> analyzer.classify_sentiment(-0.3)
            'negative'
            >>> analyzer.classify_sentiment(0.02)
            'neutral'
        """
        if score > threshold:
            return "positive"
        elif score < -threshold:
            return "negative"
        else:
            return "neutral"

    def analyze_dataframe(
        self, df: pd.DataFrame, text_column: str = "headline"
    ) -> pd.DataFrame:
        """
        Analyze sentiment for all headlines in a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing news headlines
            text_column (str): Name of column containing text (default: 'headline')

        Returns:
            pd.DataFrame: DataFrame with added sentiment columns:
                - sentiment_score: polarity score (-1 to +1)
                - sentiment_class: classification (positive/negative/neutral)

        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> news_df = pd.DataFrame({'headline': ['Good news', 'Bad news']})
            >>> result = analyzer.analyze_dataframe(news_df)
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        # Create a copy to avoid modifying original
        result_df = df.copy()

        print(f"Analyzing sentiment for {len(result_df)} headlines...")

        # Apply sentiment analysis
        result_df["sentiment_score"] = result_df[text_column].apply(
            self.analyze_headline
        )

        # Classify sentiment
        result_df["sentiment_class"] = result_df["sentiment_score"].apply(
            self.classify_sentiment
        )

        print("✅ Sentiment analysis complete")

        return result_df

    def aggregate_daily_sentiment(
        self,
        df: pd.DataFrame,
        date_column: str = "aligned_date",
        symbol_column: str = "symbol",
    ) -> pd.DataFrame:
        """
        Aggregate sentiment scores by date and stock symbol.
        Handles multiple news articles per stock per day.

        Args:
            df (pd.DataFrame): DataFrame with sentiment scores
            date_column (str): Name of date column (default: 'aligned_date')
            symbol_column (str): Name of symbol column (default: 'symbol')

        Returns:
            pd.DataFrame: Aggregated dataset with daily sentiment metrics:
                - daily_sentiment: mean sentiment score
                - sentiment_std: standard deviation of sentiment
                - article_count: number of articles
                - positive_count: number of positive articles
                - negative_count: number of negative articles
                - neutral_count: number of neutral articles

        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> daily_sentiment = analyzer.aggregate_daily_sentiment(sentiment_df)
        """
        required_columns = [
            date_column,
            symbol_column,
            "sentiment_score",
            "sentiment_class",
        ]
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Group by date and symbol
        agg_df = (
            df.groupby([symbol_column, date_column])
            .agg(
                daily_sentiment=("sentiment_score", "mean"),
                sentiment_std=("sentiment_score", "std"),
                sentiment_min=("sentiment_score", "min"),
                sentiment_max=("sentiment_score", "max"),
                article_count=("sentiment_score", "count"),
                positive_count=(
                    "sentiment_class",
                    lambda x: (x == "positive").sum(),
                ),
                negative_count=(
                    "sentiment_class",
                    lambda x: (x == "negative").sum(),
                ),
                neutral_count=(
                    "sentiment_class",
                    lambda x: (x == "neutral").sum(),
                ),
            )
            .reset_index()
        )

        # Fill NaN std with 0 (when only one article)
        agg_df["sentiment_std"] = agg_df["sentiment_std"].fillna(0)

        return agg_df

    def get_sentiment_distribution(
        self, df: pd.DataFrame, by_stock: bool = False
    ) -> Dict[str, any]:
        """
        Get distribution statistics for sentiment scores.

        Args:
            df (pd.DataFrame): DataFrame with sentiment_score column
            by_stock (bool): Whether to calculate distribution by stock (default: False)

        Returns:
            Dict[str, any]: Dictionary containing distribution statistics

        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> stats = analyzer.get_sentiment_distribution(sentiment_df)
        """
        if "sentiment_score" not in df.columns:
            raise ValueError("DataFrame must have 'sentiment_score' column")

        if not by_stock:
            stats = {
                "mean": df["sentiment_score"].mean(),
                "median": df["sentiment_score"].median(),
                "std": df["sentiment_score"].std(),
                "min": df["sentiment_score"].min(),
                "max": df["sentiment_score"].max(),
                "positive_pct": (df["sentiment_class"] == "positive").sum()
                / len(df)
                * 100,
                "negative_pct": (df["sentiment_class"] == "negative").sum()
                / len(df)
                * 100,
                "neutral_pct": (df["sentiment_class"] == "neutral").sum()
                / len(df)
                * 100,
            }
        else:
            if "symbol" not in df.columns:
                raise ValueError(
                    "DataFrame must have 'symbol' column for by_stock analysis"
                )

            stats = {}
            for symbol in df["symbol"].unique():
                symbol_df = df[df["symbol"] == symbol]
                stats[symbol] = {
                    "mean": symbol_df["sentiment_score"].mean(),
                    "median": symbol_df["sentiment_score"].median(),
                    "std": symbol_df["sentiment_score"].std(),
                    "count": len(symbol_df),
                    "positive_pct": (symbol_df["sentiment_class"] == "positive").sum()
                    / len(symbol_df)
                    * 100,
                    "negative_pct": (symbol_df["sentiment_class"] == "negative").sum()
                    / len(symbol_df)
                    * 100,
                    "neutral_pct": (symbol_df["sentiment_class"] == "neutral").sum()
                    / len(symbol_df)
                    * 100,
                }

        return stats

    def filter_by_sentiment(
        self, df: pd.DataFrame, sentiment_class: str
    ) -> pd.DataFrame:
        """
        Filter DataFrame by sentiment classification.

        Args:
            df (pd.DataFrame): DataFrame with sentiment_class column
            sentiment_class (str): Sentiment class to filter ('positive', 'negative', 'neutral')

        Returns:
            pd.DataFrame: Filtered DataFrame

        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> positive_news = analyzer.filter_by_sentiment(df, 'positive')
        """
        if "sentiment_class" not in df.columns:
            raise ValueError("DataFrame must have 'sentiment_class' column")

        valid_classes = ["positive", "negative", "neutral"]
        if sentiment_class not in valid_classes:
            raise ValueError(f"Invalid sentiment_class. Must be one of {valid_classes}")

        return df[df["sentiment_class"] == sentiment_class].copy()
