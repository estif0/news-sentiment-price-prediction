import pytest
import pandas as pd
import numpy as np
from src.core.financial_analyzer import FinancialAnalyzer, TALIB_AVAILABLE


class TestFinancialAnalyzer:
    """Test suite for FinancialAnalyzer class."""

    @pytest.fixture
    def sample_stock_data(self):
        """Create sample stock data for testing."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        np.random.seed(42)  # For reproducible results

        # Generate realistic stock price data
        initial_price = 100
        returns = np.random.normal(
            0.001, 0.02, 100
        )  # Small daily returns with volatility
        prices = [initial_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        data = pd.DataFrame(
            {
                "Date": dates,
                "Open": prices,
                "High": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                "Low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                "Close": prices,
                "Volume": np.random.randint(1000000, 10000000, 100),
            }
        )

        # Ensure OHLC relationships are correct
        data["High"] = data[["Open", "High", "Close"]].max(axis=1)
        data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

        return data.set_index("Date")

    @pytest.fixture
    def analyzer(self):
        """Create FinancialAnalyzer instance."""
        return FinancialAnalyzer()

    def test_calculate_returns(self, analyzer, sample_stock_data):
        """Test daily returns calculation."""
        returns = analyzer.calculate_returns(sample_stock_data)

        # Should have one less return than prices (first is NaN)
        assert len(returns) == len(sample_stock_data)
        assert pd.isna(returns.iloc[0])  # First return should be NaN

        # Calculate expected second return manually
        expected_second_return = (
            sample_stock_data["Close"].iloc[1] / sample_stock_data["Close"].iloc[0]
        ) - 1
        assert abs(returns.iloc[1] - expected_second_return) < 1e-10

    def test_calculate_volatility(self, analyzer, sample_stock_data):
        """Test volatility calculation."""
        volatility = analyzer.calculate_volatility(
            sample_stock_data, window=10, annualized=False
        )

        # Should have values starting from window size
        assert len(volatility) == len(sample_stock_data)
        assert all(pd.isna(volatility.iloc[:9]))  # First 9 should be NaN
        assert not pd.isna(volatility.iloc[10])  # 10th should have a value

    def test_calculate_sharpe_ratio(self, analyzer, sample_stock_data):
        """Test Sharpe ratio calculation."""
        sharpe = analyzer.calculate_sharpe_ratio(sample_stock_data)

        # Should be a float
        assert isinstance(sharpe, float)
        # Should be finite (not inf or NaN)
        assert np.isfinite(sharpe)

    def test_calculate_max_drawdown(self, analyzer, sample_stock_data):
        """Test maximum drawdown calculation."""
        drawdown_metrics = analyzer.calculate_max_drawdown(sample_stock_data)

        # Should return dict with expected keys
        expected_keys = ["max_drawdown", "drawdown_duration", "num_drawdown_periods"]
        assert all(key in drawdown_metrics for key in expected_keys)

        # Max drawdown should be positive (absolute value)
        assert drawdown_metrics["max_drawdown"] >= 0

        # Duration should be non-negative integer
        assert drawdown_metrics["drawdown_duration"] >= 0
        assert isinstance(drawdown_metrics["drawdown_duration"], int)

    def test_calculate_performance_metrics(self, analyzer, sample_stock_data):
        """Test comprehensive performance metrics calculation."""
        metrics = analyzer.calculate_performance_metrics(sample_stock_data)

        # Should return dict with expected keys
        expected_keys = [
            "total_return_pct",
            "annualized_return_pct",
            "volatility_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
            "win_rate_pct",
            "positive_days",
            "negative_days",
            "total_days",
        ]
        assert all(key in metrics for key in expected_keys)

        # All metrics should be finite numbers
        for key, value in metrics.items():
            assert np.isfinite(value), f"{key} should be finite, got {value}"

    def test_calculate_correlation_matrix(self, analyzer, sample_stock_data):
        """Test correlation matrix calculation."""
        # Create multiple stock data
        stock_data = {
            "AAPL": sample_stock_data,
            "GOOGL": sample_stock_data.copy(),  # Using same data for simplicity
        }

        corr_matrix = analyzer.calculate_correlation_matrix(stock_data)

        # Should be square matrix
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
        assert corr_matrix.shape[0] == len(stock_data)

        # Diagonal should be 1.0 (perfect self-correlation)
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), 1.0)

    def test_get_data_quality_report(self, analyzer, sample_stock_data):
        """Test data quality report generation."""
        report = analyzer.get_data_quality_report(sample_stock_data)

        # Should contain expected sections
        expected_keys = ["total_records", "date_range", "missing_values", "ohlcv_valid"]
        assert all(key in report for key in expected_keys)

        # Total records should match data length
        assert report["total_records"] == len(sample_stock_data)

        # OHLCV validation should pass for our generated data
        assert all(report["ohlcv_valid"].values())

    def test_analyze_multiple_stocks(self, analyzer, sample_stock_data):
        """Test multiple stocks analysis."""
        stock_data = {
            "AAPL": sample_stock_data,
            "GOOGL": sample_stock_data.copy(),
            "MSFT": sample_stock_data.copy(),
        }

        results = analyzer.analyze_multiple_stocks(stock_data)

        # Should return metrics for each stock
        assert len(results) == len(stock_data)
        assert all(symbol in results for symbol in stock_data.keys())

        # Each result should have performance metrics
        for symbol, metrics in results.items():
            assert "total_return_pct" in metrics
            assert "sharpe_ratio" in metrics

    def test_get_correlation_insights(self, analyzer):
        """Test correlation insights extraction."""
        # Create a simple correlation matrix
        corr_data = {"A": [1.0, 0.8, 0.3], "B": [0.8, 1.0, 0.2], "C": [0.3, 0.2, 1.0]}
        corr_matrix = pd.DataFrame(corr_data, index=["A", "B", "C"])

        insights = analyzer.get_correlation_insights(corr_matrix, threshold=0.7)

        # Should find A-B correlation (0.8 > 0.7)
        assert len(insights) == 1
        assert insights[0] == ("A", "B", 0.8)

    def test_empty_data_handling(self, analyzer):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()

        # Should handle empty data gracefully
        metrics = analyzer.calculate_performance_metrics(empty_data)
        assert metrics == {}

    def test_insufficient_data_handling(self, analyzer):
        """Test handling of insufficient data."""
        # Single row of data
        single_row = pd.DataFrame(
            {"Close": [100], "Volume": [1000000]},
            index=pd.date_range("2020-01-01", periods=1),
        )

        metrics = analyzer.calculate_performance_metrics(single_row)
        assert metrics == {}


class TestTechnicalIndicators:
    """Test suite for TA-Lib technical indicators."""

    @pytest.fixture
    def sample_stock_data(self):
        """Create sample stock data with sufficient history for indicators."""
        dates = pd.date_range("2020-01-01", periods=250, freq="D")
        np.random.seed(42)

        # Generate realistic stock price data with trend
        initial_price = 100
        trend = 0.0005  # Slight upward trend
        volatility = 0.02

        prices = [initial_price]
        for _ in range(249):
            change = np.random.normal(trend, volatility)
            prices.append(prices[-1] * (1 + change))

        data = pd.DataFrame(
            {
                "Date": dates,
                "Open": prices,
                "High": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                "Low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                "Close": prices,
                "Volume": np.random.randint(1000000, 10000000, 250),
            }
        )

        data["High"] = data[["Open", "High", "Close"]].max(axis=1)
        data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

        return data.set_index("Date")

    @pytest.fixture
    def analyzer(self):
        """Create FinancialAnalyzer instance."""
        return FinancialAnalyzer()

    @pytest.mark.skipif(not TALIB_AVAILABLE, reason="TA-Lib not installed")
    def test_calculate_sma(self, analyzer, sample_stock_data):
        """Test Simple Moving Average calculation."""
        sma = analyzer.calculate_sma(sample_stock_data, period=20)

        # Should have same length as input
        assert len(sma) == len(sample_stock_data)

        # First 19 values should be NaN (not enough data)
        assert all(pd.isna(sma.iloc[:19]))

        # 20th value should be the average of first 20 closing prices
        expected_sma = sample_stock_data["Close"].iloc[:20].mean()
        assert abs(sma.iloc[19] - expected_sma) < 0.01

        # Values should be positive
        assert all(sma.dropna() > 0)

    @pytest.mark.skipif(not TALIB_AVAILABLE, reason="TA-Lib not installed")
    def test_calculate_ema(self, analyzer, sample_stock_data):
        """Test Exponential Moving Average calculation."""
        ema = analyzer.calculate_ema(sample_stock_data, period=20)

        # Should have same length as input
        assert len(ema) == len(sample_stock_data)

        # First values will be NaN
        assert pd.isna(ema.iloc[0])

        # EMA should stabilize after sufficient data
        assert not pd.isna(ema.iloc[-1])

        # Values should be positive
        assert all(ema.dropna() > 0)

    @pytest.mark.skipif(not TALIB_AVAILABLE, reason="TA-Lib not installed")
    def test_calculate_rsi(self, analyzer, sample_stock_data):
        """Test RSI calculation."""
        rsi = analyzer.calculate_rsi(sample_stock_data, period=14)

        # Should have same length as input
        assert len(rsi) == len(sample_stock_data)

        # First values will be NaN
        assert pd.isna(rsi.iloc[0])

        # RSI values should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert all(valid_rsi >= 0)
        assert all(valid_rsi <= 100)

    @pytest.mark.skipif(not TALIB_AVAILABLE, reason="TA-Lib not installed")
    def test_calculate_macd(self, analyzer, sample_stock_data):
        """Test MACD calculation."""
        macd_data = analyzer.calculate_macd(sample_stock_data)

        # Should return DataFrame with 3 columns
        assert isinstance(macd_data, pd.DataFrame)
        assert list(macd_data.columns) == ["MACD", "Signal", "Histogram"]

        # Should have same length as input
        assert len(macd_data) == len(sample_stock_data)

        # First values will be NaN
        assert pd.isna(macd_data["MACD"].iloc[0])

        # Histogram should equal MACD - Signal
        valid_indices = macd_data.dropna().index
        if len(valid_indices) > 0:
            for idx in valid_indices[-10:]:  # Check last 10 valid values
                expected_hist = (
                    macd_data.loc[idx, "MACD"] - macd_data.loc[idx, "Signal"]
                )
                assert abs(macd_data.loc[idx, "Histogram"] - expected_hist) < 1e-6

    @pytest.mark.skipif(not TALIB_AVAILABLE, reason="TA-Lib not installed")
    def test_calculate_bollinger_bands(self, analyzer, sample_stock_data):
        """Test Bollinger Bands calculation."""
        bb_data = analyzer.calculate_bollinger_bands(sample_stock_data, period=20)

        # Should return DataFrame with 3 columns
        assert isinstance(bb_data, pd.DataFrame)
        assert list(bb_data.columns) == ["Upper", "Middle", "Lower"]

        # Should have same length as input
        assert len(bb_data) == len(sample_stock_data)

        # Upper band should be above middle, middle above lower
        valid_data = bb_data.dropna()
        if len(valid_data) > 0:
            assert all(valid_data["Upper"] >= valid_data["Middle"])
            assert all(valid_data["Middle"] >= valid_data["Lower"])

    @pytest.mark.skipif(not TALIB_AVAILABLE, reason="TA-Lib not installed")
    def test_add_technical_indicators(self, analyzer, sample_stock_data):
        """Test adding all technical indicators at once."""
        result = analyzer.add_technical_indicators(sample_stock_data)

        # Should include all original columns
        for col in sample_stock_data.columns:
            assert col in result.columns

        # Should include new indicator columns
        expected_indicators = [
            "SMA_20",
            "SMA_50",
            "SMA_200",
            "EMA_12",
            "EMA_26",
            "RSI_14",
            "MACD",
            "MACD_Signal",
            "MACD_Histogram",
            "BB_Upper",
            "BB_Middle",
            "BB_Lower",
        ]

        for indicator in expected_indicators:
            assert indicator in result.columns, f"{indicator} not found in result"

        # Should have same length as input
        assert len(result) == len(sample_stock_data)

    @pytest.mark.skipif(not TALIB_AVAILABLE, reason="TA-Lib not installed")
    def test_indicators_with_insufficient_data(self, analyzer):
        """Test indicators with insufficient data."""
        # Create small dataset
        small_data = pd.DataFrame(
            {
                "Close": [100, 101, 102, 103, 104],
                "Open": [99, 100, 101, 102, 103],
                "High": [101, 102, 103, 104, 105],
                "Low": [98, 99, 100, 101, 102],
                "Volume": [1000000] * 5,
            },
            index=pd.date_range("2020-01-01", periods=5),
        )

        # Should not raise error, but will have many NaN values
        sma = analyzer.calculate_sma(small_data, period=20)
        assert all(pd.isna(sma))  # All NaN because not enough data

    def test_talib_import_error_handling(self, analyzer):
        """Test that appropriate error is raised when TA-Lib not available."""
        if not TALIB_AVAILABLE:
            sample_data = pd.DataFrame(
                {"Close": [100, 101, 102]},
                index=pd.date_range("2020-01-01", periods=3),
            )

            with pytest.raises(ImportError, match="TA-Lib is required"):
                analyzer.calculate_sma(sample_data)
