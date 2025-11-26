# Notebooks Directory

This directory contains Jupyter notebooks for interactive data exploration, analysis, and reporting.

## Contents

### `01_eda_news.ipynb` ✅

-   **Purpose**: Comprehensive Exploratory Data Analysis (EDA) of the financial news dataset
-   **Dataset**: 1.4M+ financial news articles (2012-2020)
-   **Analysis Includes**:
    -   Text characteristics and headline analysis
    -   Publisher landscape and content specialization
    -   Temporal patterns and publication frequency
    -   Event detection and market spike analysis
    -   Data quality assessment and insights
-   **Status**: Complete (Task 1)

### `02_quantitative_analysis.ipynb` ✅

-   **Purpose**: Quantitative analysis of stock price data with technical indicators
-   **Stocks Analyzed**: AAPL, AMZN, GOOG, META, MSFT, NVDA (6 major tech stocks)
-   **Analysis Includes**:
    -   Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
    -   Financial metrics (returns, volatility, Sharpe ratio, max drawdown)
    -   Risk analysis and portfolio performance comparison
    -   Multi-stock correlation analysis
    -   Professional visualization of technical patterns
-   **Status**: Complete (Task 2)

### `03_correlation_analysis.ipynb` ✅

-   **Purpose**: Correlation analysis between news sentiment and stock price movements
-   **Dataset**: 5,064 merged news-stock records, 1,588 daily aggregations
-   **Analysis Includes**:
    -   Date alignment and dataset merging (UTC-4 timezone handling)
    -   TextBlob sentiment scoring for 1.4M+ headlines
    -   Daily sentiment aggregation by stock
    -   Pearson correlation analysis (NVDA: r=0.0992, p=0.0008)
    -   Lag analysis (T+1, T+2, T+3) for predictive insights
    -   Visualization: scatter plots, heatmaps, time series
    -   Investment strategy development
-   **Status**: Complete (Task 3)

## Usage

```bash
# Start Jupyter from project root
jupyter notebook notebooks/

# Or open specific notebooks
jupyter notebook notebooks/01_eda_news.ipynb                # Task 1: EDA
jupyter notebook notebooks/02_quantitative_analysis.ipynb   # Task 2: Technical Analysis
jupyter notebook notebooks/03_correlation_analysis.ipynb    # Task 3: Sentiment Correlation
```

## Dependencies

All notebooks use the custom analysis modules from `src/core/`:

-   `DataLoader`: Load and validate news and stock datasets
-   `EDAAnalyzer`: Perform statistical and textual analysis
-   `FinancialAnalyzer`: Calculate technical indicators and financial metrics
-   `SentimentAnalyzer`: Analyze sentiment using TextBlob NLP
-   `DataProcessor`: Align dates and merge news-stock datasets
-   `Visualizer`: Generate professional plots and charts

## Key Results

### Task 1: EDA Insights

-   1.4M+ articles with 100% completeness
-   73.12 character average headline length
-   Top 3 publishers: 42% market concentration
-   61 major market events detected

### Task 2: Technical Analysis

-   6 stocks analyzed with full OHLCV data
-   Technical indicators: SMA, EMA, RSI, MACD, BB
-   Risk metrics: Sharpe ratios, max drawdowns calculated
-   32 unit tests passing for financial analysis

### Task 3: Correlation Analysis

-   5,064 news-stock records merged successfully
-   1,588 daily sentiment-return aggregations
-   **NVDA**: r=0.0992, p=0.0008 (statistically significant)
-   **GOOG**: r=0.0849, p=0.111 (borderline significant)
-   Same-day correlations strongest (efficient markets)
-   60 total unit tests passing

## Notes

-   Notebooks are designed to run from the `notebooks/` directory
-   Auto-reload is enabled for dynamic module development
-   All outputs and visualizations are preserved for documentation
-   Each notebook corresponds to a specific project task (1, 2, 3)
-   Comprehensive markdown explanations accompany all analyses
