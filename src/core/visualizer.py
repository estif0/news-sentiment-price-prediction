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

    # ==================== Technical Indicator Visualizations ====================

    def plot_stock_with_ma(
        self,
        data: pd.DataFrame,
        symbol: str,
        ma_columns: Optional[List[str]] = None,
        price_column: str = "Close",
    ) -> None:
        """
        Plot stock price with moving average overlays.

        Args:
            data (pd.DataFrame): Stock data with price and MA columns
            symbol (str): Stock symbol for title
            ma_columns (Optional[List[str]]): List of MA column names to plot
            price_column (str): Price column to plot (default: 'Close')
        """
        if ma_columns is None:
            # Default to common MA columns if they exist
            ma_columns = [col for col in data.columns if "SMA" in col or "EMA" in col]

        plt.figure(figsize=(14, 7))

        # Plot price
        plt.plot(
            data.index,
            data[price_column],
            label=f"{symbol} {price_column}",
            linewidth=2,
            alpha=0.8,
            color="navy",
        )

        # Plot moving averages
        colors = ["red", "orange", "green", "purple", "brown"]
        for i, ma_col in enumerate(ma_columns):
            if ma_col in data.columns:
                color = colors[i % len(colors)]
                plt.plot(
                    data.index,
                    data[ma_col],
                    label=ma_col,
                    linewidth=1.5,
                    alpha=0.7,
                    linestyle="--",
                    color=color,
                )

        plt.title(
            f"{symbol} Price with Moving Averages", fontsize=14, fontweight="bold"
        )
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_rsi_indicator(
        self, data: pd.DataFrame, symbol: str, rsi_column: str = "RSI_14"
    ) -> None:
        """
        Plot RSI indicator with overbought/oversold zones.

        Args:
            data (pd.DataFrame): Stock data with RSI column
            symbol (str): Stock symbol for title
            rsi_column (str): RSI column name (default: 'RSI_14')
        """
        plt.figure(figsize=(14, 5))

        # Plot RSI
        plt.plot(data.index, data[rsi_column], label="RSI", linewidth=2, color="purple")

        # Overbought/Oversold lines
        plt.axhline(y=70, color="red", linestyle="--", linewidth=1, label="Overbought")
        plt.axhline(y=30, color="green", linestyle="--", linewidth=1, label="Oversold")
        plt.axhline(y=50, color="gray", linestyle=":", linewidth=1, alpha=0.5)

        # Fill zones
        plt.fill_between(data.index, 70, 100, alpha=0.1, color="red")
        plt.fill_between(data.index, 0, 30, alpha=0.1, color="green")

        plt.title(f"{symbol} RSI Indicator", fontsize=14, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("RSI")
        plt.ylim(0, 100)
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_macd_indicator(
        self,
        data: pd.DataFrame,
        symbol: str,
        macd_col: str = "MACD",
        signal_col: str = "MACD_Signal",
        hist_col: str = "MACD_Histogram",
    ) -> None:
        """
        Plot MACD indicator with signal line and histogram.

        Args:
            data (pd.DataFrame): Stock data with MACD columns
            symbol (str): Stock symbol for title
            macd_col (str): MACD column name (default: 'MACD')
            signal_col (str): Signal line column name (default: 'MACD_Signal')
            hist_col (str): Histogram column name (default: 'MACD_Histogram')
        """
        plt.figure(figsize=(14, 6))

        # Plot MACD and Signal lines
        plt.plot(data.index, data[macd_col], label="MACD", linewidth=2, color="blue")
        plt.plot(
            data.index,
            data[signal_col],
            label="Signal",
            linewidth=2,
            color="red",
            linestyle="--",
        )

        # Plot histogram
        colors = ["green" if val >= 0 else "red" for val in data[hist_col]]
        plt.bar(data.index, data[hist_col], label="Histogram", alpha=0.3, color=colors)

        plt.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        plt.title(f"{symbol} MACD Indicator", fontsize=14, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("MACD")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_technical_analysis_dashboard(
        self, data: pd.DataFrame, symbol: str, price_column: str = "Close"
    ) -> None:
        """
        Create comprehensive 3-panel technical analysis dashboard.

        Args:
            data (pd.DataFrame): Stock data with technical indicators
            symbol (str): Stock symbol for title
            price_column (str): Price column to plot (default: 'Close')
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(
            f"{symbol} Technical Analysis Dashboard",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        # Panel 1: Price with Moving Averages
        axes[0].plot(
            data.index,
            data[price_column],
            label=f"{symbol} Price",
            linewidth=2,
            color="navy",
        )

        # Plot MAs if available
        ma_cols = [col for col in data.columns if "SMA" in col or "EMA" in col]
        colors = ["red", "orange", "green", "purple", "brown"]
        for i, ma_col in enumerate(ma_cols[:5]):  # Limit to 5 MAs
            if ma_col in data.columns:
                axes[0].plot(
                    data.index,
                    data[ma_col],
                    label=ma_col,
                    linewidth=1.5,
                    alpha=0.7,
                    linestyle="--",
                    color=colors[i % len(colors)],
                )

        axes[0].set_title("Price & Moving Averages", fontweight="bold")
        axes[0].set_ylabel("Price ($)")
        axes[0].legend(loc="best", fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Panel 2: RSI
        rsi_col = "RSI_14" if "RSI_14" in data.columns else None
        if rsi_col:
            axes[1].plot(
                data.index, data[rsi_col], linewidth=2, color="purple", label="RSI"
            )
            axes[1].axhline(y=70, color="red", linestyle="--", linewidth=1, alpha=0.7)
            axes[1].axhline(y=30, color="green", linestyle="--", linewidth=1, alpha=0.7)
            axes[1].axhline(y=50, color="gray", linestyle=":", linewidth=1, alpha=0.5)
            axes[1].fill_between(data.index, 70, 100, alpha=0.1, color="red")
            axes[1].fill_between(data.index, 0, 30, alpha=0.1, color="green")
            axes[1].set_ylim(0, 100)
            axes[1].set_title("RSI Indicator", fontweight="bold")
            axes[1].set_ylabel("RSI")
            axes[1].legend(loc="best")
            axes[1].grid(True, alpha=0.3)

        # Panel 3: MACD
        macd_col = "MACD" if "MACD" in data.columns else None
        signal_col = "MACD_Signal" if "MACD_Signal" in data.columns else None
        hist_col = "MACD_Histogram" if "MACD_Histogram" in data.columns else None

        if macd_col and signal_col and hist_col:
            axes[2].plot(
                data.index, data[macd_col], linewidth=2, color="blue", label="MACD"
            )
            axes[2].plot(
                data.index,
                data[signal_col],
                linewidth=2,
                color="red",
                linestyle="--",
                label="Signal",
            )

            colors = ["green" if val >= 0 else "red" for val in data[hist_col]]
            axes[2].bar(data.index, data[hist_col], alpha=0.3, color=colors)

            axes[2].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            axes[2].set_title("MACD Indicator", fontweight="bold")
            axes[2].set_ylabel("MACD")
            axes[2].set_xlabel("Date")
            axes[2].legend(loc="best")
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_bollinger_bands(
        self,
        data: pd.DataFrame,
        symbol: str,
        price_column: str = "Close",
        bb_upper: str = "BB_Upper",
        bb_middle: str = "BB_Middle",
        bb_lower: str = "BB_Lower",
    ) -> None:
        """
        Plot Bollinger Bands with price.

        Args:
            data (pd.DataFrame): Stock data with Bollinger Band columns
            symbol (str): Stock symbol for title
            price_column (str): Price column (default: 'Close')
            bb_upper (str): Upper band column (default: 'BB_Upper')
            bb_middle (str): Middle band column (default: 'BB_Middle')
            bb_lower (str): Lower band column (default: 'BB_Lower')
        """
        plt.figure(figsize=(14, 7))

        # Plot price
        plt.plot(
            data.index,
            data[price_column],
            label=f"{symbol} Price",
            linewidth=2,
            color="navy",
        )

        # Plot Bollinger Bands
        if all(col in data.columns for col in [bb_upper, bb_middle, bb_lower]):
            plt.plot(
                data.index,
                data[bb_upper],
                label="Upper Band",
                linewidth=1.5,
                color="red",
                linestyle="--",
                alpha=0.7,
            )
            plt.plot(
                data.index,
                data[bb_middle],
                label="Middle Band (SMA)",
                linewidth=1.5,
                color="orange",
                linestyle="--",
                alpha=0.7,
            )
            plt.plot(
                data.index,
                data[bb_lower],
                label="Lower Band",
                linewidth=1.5,
                color="green",
                linestyle="--",
                alpha=0.7,
            )

            # Fill between bands
            plt.fill_between(
                data.index, data[bb_upper], data[bb_lower], alpha=0.1, color="gray"
            )

        plt.title(f"{symbol} Bollinger Bands", fontsize=14, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
