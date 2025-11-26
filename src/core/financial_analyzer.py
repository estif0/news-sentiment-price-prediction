import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available. Technical indicators will not work.")

try:
    import pynance as pn

    PYNANCE_AVAILABLE = True
except ImportError:
    PYNANCE_AVAILABLE = False
    print(
        "Warning: PyNance not available. Some financial calculations will use fallback methods."
    )


class FinancialAnalyzer:
    """
    Class for performing financial analysis on stock data including technical indicators,
    risk metrics, and performance calculations.
    """

    def __init__(self):
        """Initialize the FinancialAnalyzer."""
        if not TALIB_AVAILABLE:
            print(
                "Warning: TA-Lib is not installed. Technical indicators may not work."
            )
        if not PYNANCE_AVAILABLE:
            print(
                "Warning: PyNance is not installed. Some financial calculations will use fallback methods."
            )

    # ==================== Technical Indicators (TA-Lib) ====================

    def calculate_sma(
        self, data: pd.DataFrame, column: str = "Close", period: int = 20
    ) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA) using TA-Lib.

        SMA = Sum of closing prices over period / period

        Args:
            data (pd.DataFrame): Stock data
            column (str): Column to calculate SMA for (default: 'Close')
            period (int): SMA period (default: 20)

        Returns:
            pd.Series: Simple Moving Average values

        Raises:
            ImportError: If TA-Lib is not installed
        """
        if not TALIB_AVAILABLE:
            raise ImportError(
                "TA-Lib is required for technical indicators. "
                "Install it with: pip install TA-Lib"
            )

        prices = data[column].astype(np.float64).values
        sma = talib.SMA(prices, timeperiod=period)
        return pd.Series(sma, index=data.index, name=f"SMA_{period}")

    def calculate_ema(
        self, data: pd.DataFrame, column: str = "Close", period: int = 20
    ) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA) using TA-Lib.

        EMA gives more weight to recent prices, making it more responsive
        to price changes than SMA.

        Formula: EMA = (Close - EMA(previous)) × multiplier + EMA(previous)
        Where multiplier = 2 / (period + 1)

        Args:
            data (pd.DataFrame): Stock data
            column (str): Column to calculate EMA for (default: 'Close')
            period (int): EMA period (default: 20)

        Returns:
            pd.Series: Exponential Moving Average values

        Raises:
            ImportError: If TA-Lib is not installed
        """
        if not TALIB_AVAILABLE:
            raise ImportError(
                "TA-Lib is required for technical indicators. "
                "Install it with: pip install TA-Lib"
            )

        prices = data[column].astype(np.float64).values
        ema = talib.EMA(prices, timeperiod=period)
        return pd.Series(ema, index=data.index, name=f"EMA_{period}")

    def calculate_rsi(
        self, data: pd.DataFrame, column: str = "Close", period: int = 14
    ) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI) using TA-Lib.

        RSI measures the magnitude of recent price changes to evaluate
        overbought or oversold conditions.

        Formula: RSI = 100 - (100 / (1 + RS))
        Where RS = Average Gain / Average Loss over period

        Interpretation:
        - RSI > 70: Overbought (potential sell signal)
        - RSI < 30: Oversold (potential buy signal)
        - RSI = 50: Neutral

        Args:
            data (pd.DataFrame): Stock data
            column (str): Column to calculate RSI for (default: 'Close')
            period (int): RSI period (default: 14)

        Returns:
            pd.Series: RSI values (0-100)

        Raises:
            ImportError: If TA-Lib is not installed
        """
        if not TALIB_AVAILABLE:
            raise ImportError(
                "TA-Lib is required for technical indicators. "
                "Install it with: pip install TA-Lib"
            )

        prices = data[column].astype(np.float64).values
        rsi = talib.RSI(prices, timeperiod=period)
        return pd.Series(rsi, index=data.index, name=f"RSI_{period}")

    def calculate_macd(
        self,
        data: pd.DataFrame,
        column: str = "Close",
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> pd.DataFrame:
        """
        Calculate Moving Average Convergence Divergence (MACD) using TA-Lib.

        MACD is a trend-following momentum indicator that shows the relationship
        between two moving averages of a security's price.

        Components:
        - MACD Line: (12-period EMA - 26-period EMA)
        - Signal Line: 9-period EMA of MACD Line
        - MACD Histogram: MACD Line - Signal Line

        Interpretation:
        - MACD > Signal: Bullish (buy signal)
        - MACD < Signal: Bearish (sell signal)
        - Histogram crossing zero: Potential trend change

        Args:
            data (pd.DataFrame): Stock data
            column (str): Column to calculate MACD for (default: 'Close')
            fast_period (int): Fast EMA period (default: 12)
            slow_period (int): Slow EMA period (default: 26)
            signal_period (int): Signal line period (default: 9)

        Returns:
            pd.DataFrame: DataFrame with columns ['MACD', 'Signal', 'Histogram']

        Raises:
            ImportError: If TA-Lib is not installed
        """
        if not TALIB_AVAILABLE:
            raise ImportError(
                "TA-Lib is required for technical indicators. "
                "Install it with: pip install TA-Lib"
            )

        prices = data[column].astype(np.float64).values
        macd, signal, histogram = talib.MACD(
            prices,
            fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period,
        )

        return pd.DataFrame(
            {"MACD": macd, "Signal": signal, "Histogram": histogram}, index=data.index
        )

    def calculate_bollinger_bands(
        self,
        data: pd.DataFrame,
        column: str = "Close",
        period: int = 20,
        nbdevup: int = 2,
        nbdevdn: int = 2,
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands using TA-Lib.

        Bollinger Bands consist of a middle band (SMA) and two outer bands
        (standard deviations away from the middle band).

        Args:
            data (pd.DataFrame): Stock data
            column (str): Column to calculate bands for (default: 'Close')
            period (int): Period for SMA and standard deviation (default: 20)
            nbdevup (int): Number of standard deviations for upper band (default: 2)
            nbdevdn (int): Number of standard deviations for lower band (default: 2)

        Returns:
            pd.DataFrame: DataFrame with columns ['Upper', 'Middle', 'Lower']

        Raises:
            ImportError: If TA-Lib is not installed
        """
        if not TALIB_AVAILABLE:
            raise ImportError(
                "TA-Lib is required for technical indicators. "
                "Install it with: pip install TA-Lib"
            )

        prices = data[column].astype(np.float64).values
        upper, middle, lower = talib.BBANDS(
            prices, timeperiod=period, nbdevup=nbdevup, nbdevdn=nbdevdn
        )

        return pd.DataFrame(
            {"Upper": upper, "Middle": middle, "Lower": lower}, index=data.index
        )

    def add_technical_indicators(
        self, data: pd.DataFrame, column: str = "Close"
    ) -> pd.DataFrame:
        """
        Add all technical indicators to the DataFrame.

        Args:
            data (pd.DataFrame): Stock data
            column (str): Column to calculate indicators for (default: 'Close')

        Returns:
            pd.DataFrame: Original data with technical indicator columns added
        """
        result = data.copy()

        # Moving Averages
        result["SMA_20"] = self.calculate_sma(data, column, 20)
        result["SMA_50"] = self.calculate_sma(data, column, 50)
        result["SMA_200"] = self.calculate_sma(data, column, 200)
        result["EMA_12"] = self.calculate_ema(data, column, 12)
        result["EMA_26"] = self.calculate_ema(data, column, 26)

        # RSI
        result["RSI_14"] = self.calculate_rsi(data, column, 14)

        # MACD
        macd_data = self.calculate_macd(data, column)
        result["MACD"] = macd_data["MACD"]
        result["MACD_Signal"] = macd_data["Signal"]
        result["MACD_Histogram"] = macd_data["Histogram"]

        # Bollinger Bands
        bb_data = self.calculate_bollinger_bands(data, column)
        result["BB_Upper"] = bb_data["Upper"]
        result["BB_Middle"] = bb_data["Middle"]
        result["BB_Lower"] = bb_data["Lower"]

        return result

    # ==================== Existing Methods ====================

    def calculate_returns(self, data: pd.DataFrame, column: str = "Close") -> pd.Series:
        """
        Calculate daily percentage returns using PyNance when available.

        Args:
            data (pd.DataFrame): Stock data with price information
            column (str): Column name to calculate returns for (default: 'Close')

        Returns:
            pd.Series: Daily percentage returns
        """
        if PYNANCE_AVAILABLE:
            try:
                temp_df = pd.DataFrame({column: data[column]})
                returns_df = pn.tech.ret(temp_df, selection=column)
                result = pd.Series(
                    index=data.index, dtype=float, name=f"{column}_returns"
                )
                result.iloc[0] = np.nan
                result.iloc[1:] = returns_df["Return"].values
                return result
            except (KeyError, AttributeError, ValueError):
                # Fallback if PyNance doesn't work with our data structure
                return data[column].pct_change()
        else:
            # Fallback to pandas pct_change
            return data[column].pct_change()

    def calculate_volatility(
        self,
        data: pd.DataFrame,
        window: int = 20,
        column: str = "Close",
        annualized: bool = True,
    ) -> pd.Series:
        """
        Calculate rolling volatility using PyNance when available.

        Args:
            data (pd.DataFrame): Stock data
            window (int): Rolling window size (default: 20)
            column (str): Column to calculate volatility for (default: 'Close')
            annualized (bool): Whether to annualize volatility (default: True)

        Returns:
            pd.Series: Rolling volatility
        """
        if PYNANCE_AVAILABLE:
            try:
                # PyNance volatility function expects DataFrame
                temp_df = pd.DataFrame({column: data[column]})
                vol_df = pn.tech.volatility(temp_df, window=window, selection=column)
                # Extract volatility values and create Series
                vol = pd.Series(
                    (
                        vol_df["Volatility"].values
                        if "Volatility" in vol_df.columns
                        else vol_df.iloc[:, 0].values
                    ),
                    index=data.index,
                    name=f"{column}_volatility",
                )
                if annualized:
                    vol = vol * np.sqrt(252)  # Annualize if requested
                return vol
            except (KeyError, AttributeError, ValueError):
                # Fallback to manual calculation if PyNance fails
                pass

        # Fallback to manual calculation
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
