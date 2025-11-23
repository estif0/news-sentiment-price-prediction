import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional


class Visualizer:
    """
    Class for visualizing financial news data and analysis results.
    """

    def __init__(self):
        """
        Initialize the Visualizer.
        Sets the style for seaborn plots.
        """
        sns.set_theme(style="whitegrid")

    def plot_headline_length_distribution(self, lengths: pd.Series, bins: int = 30):
        """
        Plots the distribution of headline lengths.

        Args:
            lengths (pd.Series): Series containing headline lengths.
            bins (int): Number of bins for the histogram.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(lengths, bins=bins, kde=True)
        plt.title("Distribution of Headline Lengths")
        plt.xlabel("Length (characters)")
        plt.ylabel("Frequency")
        plt.show()

    def plot_top_publishers(self, publisher_counts: pd.Series, top_n: int = 10):
        """
        Plots the top N publishers by article count.

        Args:
            publisher_counts (pd.Series): Series with publisher names as index and counts as values.
            top_n (int): Number of top publishers to plot.
        """
        top_publishers = publisher_counts.head(top_n)
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x=top_publishers.values,
            y=top_publishers.index,
            hue=top_publishers.index,
            palette="viridis",
            legend=False,
        )
        plt.title(f"Top {top_n} Publishers by Article Count")
        plt.xlabel("Number of Articles")
        plt.ylabel("Publisher")
        plt.show()

    def plot_common_keywords(self, keywords: List[Tuple[str, int]]):
        """
        Plots the most common keywords.

        Args:
            keywords (List[Tuple[str, int]]): List of tuples (keyword, count).
        """
        words, counts = zip(*keywords)
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x=list(counts),
            y=list(words),
            hue=list(words),
            palette="magma",
            legend=False,
        )
        plt.title("Top Common Keywords in Headlines")
        plt.xlabel("Frequency")
        plt.ylabel("Keyword")
        plt.show()

    def plot_publication_frequency(
        self, frequency: pd.Series, title: str = "Article Publication Frequency"
    ):
        """
        Plots the publication frequency over time.

        Args:
            frequency (pd.Series): Series with date index and count values.
            title (str): Title of the plot.
        """
        plt.figure(figsize=(14, 7))
        frequency.plot()
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Number of Articles")
        plt.show()

    def plot_publishing_times(self, hourly_counts: pd.Series):
        """
        Plots the distribution of publishing times (by hour).

        Args:
            hourly_counts (pd.Series): Series with hour index and count values.
        """
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x=hourly_counts.index,
            y=hourly_counts.values,
            hue=hourly_counts.index,
            palette="coolwarm",
            legend=False,
        )
        plt.title("Article Publication by Hour of Day (UTC)")
        plt.xlabel("Hour of Day")
        plt.ylabel("Number of Articles")
        plt.show()

    # Stock Analysis Visualization Methods

    def plot_stock_overview(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Creates a comprehensive 4-panel stock analysis overview.

        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            symbol (str): Stock symbol for title
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"{symbol} Stock Analysis - Comprehensive Overview",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Price Chart
        axes[0, 0].plot(
            data.index, data["Close"], linewidth=1.5, color="navy", alpha=0.8
        )
        axes[0, 0].set_title(f"{symbol} Closing Price Over Time", fontweight="bold")
        axes[0, 0].set_xlabel("Date")
        axes[0, 0].set_ylabel("Price ($)")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 2. Volume Chart
        axes[0, 1].bar(data.index, data["Volume"], width=1, alpha=0.7, color="green")
        axes[0, 1].set_title(f"{symbol} Trading Volume", fontweight="bold")
        axes[0, 1].set_xlabel("Date")
        axes[0, 1].set_ylabel("Volume")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Returns Distribution
        daily_returns = data["Close"].pct_change().dropna()
        axes[1, 0].hist(
            daily_returns, bins=50, alpha=0.7, color="purple", edgecolor="black"
        )
        axes[1, 0].axvline(
            daily_returns.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {daily_returns.mean():.4f}",
        )
        axes[1, 0].set_title(f"{symbol} Daily Returns Distribution", fontweight="bold")
        axes[1, 0].set_xlabel("Daily Return")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. OHLC Chart (last 60 days)
        recent_data = data.tail(60)
        for i, (date, row) in enumerate(recent_data.iterrows()):
            color = "green" if row["Close"] >= row["Open"] else "red"
            # High-Low line
            axes[1, 1].plot(
                [i, i], [row["Low"], row["High"]], color="black", linewidth=1
            )
            # Open-Close rectangle
            height = abs(row["Close"] - row["Open"])
            bottom = min(row["Open"], row["Close"])
            axes[1, 1].bar(i, height, bottom=bottom, width=0.8, color=color, alpha=0.7)

        axes[1, 1].set_title(f"{symbol} OHLC Chart (Last 60 Days)", fontweight="bold")
        axes[1, 1].set_xlabel("Trading Days")
        axes[1, 1].set_ylabel("Price ($)")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_multi_stock_comparison(
        self, stock_data: Dict[str, pd.DataFrame], symbols: Optional[List[str]] = None
    ) -> None:
        """
        Creates comparative analysis plots for multiple stocks.

        Args:
            stock_data (Dict[str, pd.DataFrame]): Dictionary of stock DataFrames
            symbols (Optional[List[str]]): List of symbols to plot (default: first 4)
        """
        if symbols is None:
            symbols = list(stock_data.keys())[:4]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Multi-Stock Comparative Analysis", fontsize=16, fontweight="bold")

        # 1. Normalized Price Performance
        for symbol in symbols:
            data = stock_data[symbol]
            normalized_price = (data["Close"] / data["Close"].iloc[0]) * 100
            axes[0, 0].plot(
                data.index, normalized_price, label=symbol, linewidth=2, alpha=0.8
            )

        axes[0, 0].set_title(
            "Normalized Price Performance (Base=100)", fontweight="bold"
        )
        axes[0, 0].set_xlabel("Date")
        axes[0, 0].set_ylabel("Normalized Price")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 2. Volume Comparison (Smoothed Average)
        for symbol in symbols:
            data = stock_data[symbol]

            # Handle different index types
            if isinstance(data.index, pd.DatetimeIndex):
                # Use monthly resampling for datetime index
                volume_data = data["Volume"].resample("M").mean()
                x_values = volume_data.index
                y_values = volume_data.values
            else:
                # Use rolling window average for non-datetime index
                window_size = max(1, len(data) // 50)  # Approximately 50 points
                volume_data = data["Volume"].rolling(window=window_size).mean()
                # Sample every window_size points for smoother visualization
                x_values = data.index[::window_size]
                y_values = volume_data.iloc[::window_size].values

            axes[0, 1].plot(
                x_values,
                y_values,
                label=symbol,
                linewidth=2,
                alpha=0.8,
            )

        axes[0, 1].set_title("Average Trading Volume (Smoothed)", fontweight="bold")
        axes[0, 1].set_xlabel("Date")
        axes[0, 1].set_ylabel("Average Volume")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis="x", rotation=45)

        # 3. Volatility Comparison
        for symbol in symbols:
            data = stock_data[symbol]
            daily_returns = data["Close"].pct_change()
            volatility = daily_returns.rolling(window=30).std() * np.sqrt(252) * 100
            axes[1, 0].plot(
                data.index, volatility, label=symbol, linewidth=2, alpha=0.8
            )

        axes[1, 0].set_title(
            "30-Day Rolling Volatility (Annualized %)", fontweight="bold"
        )
        axes[1, 0].set_xlabel("Date")
        axes[1, 0].set_ylabel("Volatility (%)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis="x", rotation=45)

        # 4. Correlation Heatmap
        close_prices = pd.DataFrame()
        for symbol in symbols:
            close_prices[symbol] = stock_data[symbol]["Close"]

        correlation_matrix = close_prices.corr()
        im = axes[1, 1].imshow(
            correlation_matrix.values, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1
        )

        # Add correlation values as text
        for i in range(len(symbols)):
            for j in range(len(symbols)):
                axes[1, 1].text(
                    j,
                    i,
                    f"{correlation_matrix.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )

        axes[1, 1].set_title("Price Correlation Matrix", fontweight="bold")
        axes[1, 1].set_xticks(range(len(symbols)))
        axes[1, 1].set_yticks(range(len(symbols)))
        axes[1, 1].set_xticklabels(symbols)
        axes[1, 1].set_yticklabels(symbols)

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        cbar.set_label("Correlation Coefficient")

        plt.tight_layout()
        plt.show()

    def plot_performance_metrics(
        self, performance_data: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Visualize performance metrics for multiple stocks.

        Args:
            performance_data (Dict[str, Dict[str, float]]): Performance metrics by symbol
        """
        symbols = list(performance_data.keys())
        metrics = [
            "total_return_pct",
            "volatility_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
        ]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Performance Metrics Comparison", fontsize=16, fontweight="bold")

        # Total Returns
        returns = [
            performance_data[symbol].get("total_return_pct", 0) for symbol in symbols
        ]
        axes[0, 0].bar(symbols, returns, color="steelblue", alpha=0.7)
        axes[0, 0].set_title("Total Returns (%)", fontweight="bold")
        axes[0, 0].set_ylabel("Return (%)")
        axes[0, 0].grid(True, alpha=0.3)

        # Volatility
        volatilities = [
            performance_data[symbol].get("volatility_pct", 0) for symbol in symbols
        ]
        axes[0, 1].bar(symbols, volatilities, color="orange", alpha=0.7)
        axes[0, 1].set_title("Annualized Volatility (%)", fontweight="bold")
        axes[0, 1].set_ylabel("Volatility (%)")
        axes[0, 1].grid(True, alpha=0.3)

        # Sharpe Ratio
        sharpe_ratios = [
            performance_data[symbol].get("sharpe_ratio", 0) for symbol in symbols
        ]
        axes[1, 0].bar(symbols, sharpe_ratios, color="green", alpha=0.7)
        axes[1, 0].set_title("Sharpe Ratio", fontweight="bold")
        axes[1, 0].set_ylabel("Sharpe Ratio")
        axes[1, 0].grid(True, alpha=0.3)

        # Max Drawdown
        drawdowns = [
            performance_data[symbol].get("max_drawdown_pct", 0) for symbol in symbols
        ]
        axes[1, 1].bar(symbols, drawdowns, color="red", alpha=0.7)
        axes[1, 1].set_title("Maximum Drawdown (%)", fontweight="bold")
        axes[1, 1].set_ylabel("Drawdown (%)")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Stock Price Correlation Matrix",
    ) -> None:
        """
        Create a correlation heatmap visualization.

        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix to visualize
            title (str): Plot title
        """
        plt.figure(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".3f",
            cbar_kws={"shrink": 0.8},
        )

        plt.title(title, fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.show()
