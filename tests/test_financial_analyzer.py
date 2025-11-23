import pytest
import pandas as pd
import numpy as np
from src.core.financial_analyzer import FinancialAnalyzer


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
