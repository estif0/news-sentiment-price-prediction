import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class FinancialAnalyzer:
    """
    Class for performing financial analysis on stock data including technical indicators,
    risk metrics, and performance calculations.
    """

    def __init__(self):
        """Initialize the FinancialAnalyzer."""
        pass

    def calculate_returns(self, data: pd.DataFrame, column: str = "Close") -> pd.Series:
        """
        Calculate daily percentage returns.

        Args:
            data (pd.DataFrame): Stock data with price information
            column (str): Column name to calculate returns for (default: 'Close')

        Returns:
            pd.Series: Daily percentage returns
        """
        return data[column].pct_change()

    def calculate_volatility(
        self,
        data: pd.DataFrame,
        window: int = 20,
        column: str = "Close",
        annualized: bool = True,
    ) -> pd.Series:
        """
        Calculate rolling volatility.

        Args:
            data (pd.DataFrame): Stock data
            window (int): Rolling window size (default: 20)
            column (str): Column to calculate volatility for (default: 'Close')
            annualized (bool): Whether to annualize volatility (default: True)

        Returns:
            pd.Series: Rolling volatility
        """
        returns = self.calculate_returns(data, column)
        volatility = returns.rolling(window=window).std()

        if annualized:
            volatility = volatility * np.sqrt(252)  # Assuming 252 trading days per year

        return volatility

    def calculate_sharpe_ratio(
        self, data: pd.DataFrame, risk_free_rate: float = 0.02, column: str = "Close"
    ) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            data (pd.DataFrame): Stock data
            risk_free_rate (float): Annual risk-free rate (default: 2%)
            column (str): Column to calculate Sharpe ratio for (default: 'Close')

        Returns:
            float: Sharpe ratio
        """
        returns = self.calculate_returns(data, column).dropna()

        if len(returns) == 0:
            return 0.0

        daily_rf_rate = risk_free_rate / 252
        excess_returns = returns - daily_rf_rate

        if excess_returns.std() == 0:
            return 0.0

        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

    def calculate_max_drawdown(
        self, data: pd.DataFrame, column: str = "Close"
    ) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.

        Args:
            data (pd.DataFrame): Stock data
            column (str): Column to calculate drawdown for (default: 'Close')

        Returns:
            Dict[str, float]: Dictionary with max_drawdown, drawdown_duration, recovery_time
        """
        prices = data[column]

        # Calculate running maximum (peak)
        running_max = prices.expanding().max()

        # Calculate drawdown
        drawdown = (prices - running_max) / running_max

        # Find maximum drawdown
        max_drawdown = drawdown.min()

        # Find drawdown periods
        is_drawdown = drawdown < 0
        drawdown_periods = []
        start = None

        for i, in_dd in enumerate(is_drawdown):
            if in_dd and start is None:
                start = i
            elif not in_dd and start is not None:
                drawdown_periods.append((start, i - 1))
                start = None

        # Handle case where drawdown continues to end
        if start is not None:
            drawdown_periods.append((start, len(is_drawdown) - 1))

        # Find longest drawdown period
        max_duration = (
            0
            if not drawdown_periods
            else max(end - start + 1 for start, end in drawdown_periods)
        )

        return {
            "max_drawdown": abs(max_drawdown),
            "drawdown_duration": max_duration,
            "num_drawdown_periods": len(drawdown_periods),
        }

    def calculate_performance_metrics(
        self, data: pd.DataFrame, column: str = "Close"
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.

        Args:
            data (pd.DataFrame): Stock data
            column (str): Column to analyze (default: 'Close')

        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        if len(data) < 2:
            return {}

        returns = self.calculate_returns(data, column).dropna()

        # Basic metrics
        total_return = (data[column].iloc[-1] / data[column].iloc[0] - 1) * 100
        annual_return = (
            (data[column].iloc[-1] / data[column].iloc[0]) ** (252 / len(data)) - 1
        ) * 100

        # Risk metrics
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = self.calculate_sharpe_ratio(data, column=column)

        # Drawdown metrics
        drawdown_metrics = self.calculate_max_drawdown(data, column)

        # Additional metrics
        positive_days = (returns > 0).sum()
        negative_days = (returns < 0).sum()
        win_rate = positive_days / len(returns) * 100 if len(returns) > 0 else 0

        return {
            "total_return_pct": total_return,
            "annualized_return_pct": annual_return,
            "volatility_pct": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": drawdown_metrics["max_drawdown"] * 100,
            "win_rate_pct": win_rate,
            "positive_days": positive_days,
            "negative_days": negative_days,
            "total_days": len(returns),
        }

    def calculate_correlation_matrix(
        self, stock_data: Dict[str, pd.DataFrame], column: str = "Close"
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple stocks.

        Args:
            stock_data (Dict[str, pd.DataFrame]): Dictionary of stock DataFrames
            column (str): Column to calculate correlations for (default: 'Close')

        Returns:
            pd.DataFrame: Correlation matrix
        """
        # Align data by date
        price_data = pd.DataFrame()

        for symbol, data in stock_data.items():
            if "Date" in data.columns:
                temp_data = data.set_index("Date")
            else:
                temp_data = data
            price_data[symbol] = temp_data[column]

        return price_data.corr()

    def calculate_beta(
        self,
        stock_data: pd.DataFrame,
        market_data: pd.DataFrame,
        stock_column: str = "Close",
        market_column: str = "Close",
    ) -> float:
        """
        Calculate beta coefficient relative to market.

        Args:
            stock_data (pd.DataFrame): Individual stock data
            market_data (pd.DataFrame): Market benchmark data
            stock_column (str): Stock price column (default: 'Close')
            market_column (str): Market price column (default: 'Close')

        Returns:
            float: Beta coefficient
        """
        stock_returns = self.calculate_returns(stock_data, stock_column).dropna()
        market_returns = self.calculate_returns(market_data, market_column).dropna()

        # Align returns by index
        aligned_data = pd.concat([stock_returns, market_returns], axis=1, join="inner")
        aligned_data.columns = ["stock", "market"]

        if len(aligned_data) < 2:
            return np.nan

        # Calculate beta using covariance
        covariance = aligned_data["stock"].cov(aligned_data["market"])
        market_variance = aligned_data["market"].var()

        if market_variance == 0:
            return np.nan

        return covariance / market_variance

    def analyze_portfolio_performance(
        self,
        stock_data: Dict[str, pd.DataFrame],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Analyze portfolio performance with given weights.

        Args:
            stock_data (Dict[str, pd.DataFrame]): Dictionary of stock DataFrames
            weights (Optional[Dict[str, float]]): Portfolio weights (equal if None)

        Returns:
            Dict[str, float]: Portfolio performance metrics
        """
        if weights is None:
            # Equal weights
            weights = {symbol: 1.0 / len(stock_data) for symbol in stock_data.keys()}

        # Calculate portfolio returns
        portfolio_returns = pd.Series(dtype=float)

        for symbol, data in stock_data.items():
            if symbol in weights:
                stock_returns = self.calculate_returns(data, "Close")
                weighted_returns = stock_returns * weights[symbol]

                if portfolio_returns.empty:
                    portfolio_returns = weighted_returns
                else:
                    portfolio_returns = portfolio_returns.add(
                        weighted_returns, fill_value=0
                    )

        # Create temporary DataFrame for portfolio analysis
        portfolio_prices = (1 + portfolio_returns.fillna(0)).cumprod()
        portfolio_df = pd.DataFrame({"Close": portfolio_prices})

        return self.calculate_performance_metrics(portfolio_df, "Close")

    def get_data_quality_report(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Generate data quality report for stock data.

        Args:
            data (pd.DataFrame): Stock data to analyze

        Returns:
            Dict[str, any]: Data quality metrics
        """
        report = {}

        # Basic info
        report["total_records"] = len(data)
        report["date_range"] = {
            "start": data.index.min() if not data.index.empty else None,
            "end": data.index.max() if not data.index.empty else None,
        }

        # Missing values
        report["missing_values"] = data.isnull().sum().to_dict()
        report["missing_percentage"] = (data.isnull().sum() / len(data) * 100).to_dict()

        # OHLCV validation
        if all(
            col in data.columns for col in ["Open", "High", "Low", "Close", "Volume"]
        ):
            report["ohlcv_valid"] = {
                "high_gte_low": (data["High"] >= data["Low"]).all(),
                "high_gte_open": (data["High"] >= data["Open"]).all(),
                "high_gte_close": (data["High"] >= data["Close"]).all(),
                "low_lte_open": (data["Low"] <= data["Open"]).all(),
                "low_lte_close": (data["Low"] <= data["Close"]).all(),
                "volume_non_negative": (data["Volume"] >= 0).all(),
                "prices_positive": (
                    (data[["Open", "High", "Low", "Close"]] > 0).all()
                ).all(),
            }

        # Data gaps (for time series)
        if not data.index.empty:
            date_diffs = pd.Series(data.index).diff()
            report["max_gap_days"] = (
                date_diffs.max().days if hasattr(date_diffs.max(), "days") else 0
            )

        return report

    def analyze_multiple_stocks(
        self, stock_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform analysis on multiple stocks.

        Args:
            stock_data (Dict[str, pd.DataFrame]): Dictionary of stock DataFrames

        Returns:
            Dict[str, Dict[str, float]]: Performance metrics for each stock
        """
        results = {}

        for symbol, data in stock_data.items():
            try:
                results[symbol] = self.calculate_performance_metrics(data)
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue

        return results

    def get_correlation_insights(
        self, correlation_matrix: pd.DataFrame, threshold: float = 0.7
    ) -> List[Tuple[str, str, float]]:
        """
        Extract high correlation pairs from correlation matrix.

        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix
            threshold (float): Minimum correlation to report

        Returns:
            List[Tuple[str, str, float]]: List of (symbol1, symbol2, correlation) tuples
        """
        high_correlations = []
        symbols = correlation_matrix.index.tolist()

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_correlations.append((symbols[i], symbols[j], corr_value))

        # Sort by absolute correlation value
        high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

        return high_correlations
