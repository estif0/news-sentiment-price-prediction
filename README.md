# 📈 News Sentiment Price Prediction

> **10 Academy Week 1 Challenge**: Predicting stock price movements through financial news sentiment analysis

This project analyzes financial news sentiment to predict stock price movements, combining **Data Engineering**, **Financial Analytics**, and **Machine Learning** techniques to discover correlations between news sentiment and market behavior.

## 🎯 Project Overview

### Business Objective

**Nova Financial Solutions** aims to enhance predictive analytics capabilities by analyzing the relationship between financial news sentiment and stock market movements. This project develops a comprehensive framework for:

-   **Sentiment Analysis**: Quantifying emotional tone in financial news headlines using NLP
-   **Correlation Analysis**: Establishing statistical relationships between news sentiment and stock price movements
-   **Predictive Modeling**: Leveraging sentiment insights for investment strategy recommendations

### Key Goals

-   **Primary**: Correlate news sentiment (headline text) with stock price movements
-   **Secondary**: Identify publication patterns, publisher biases, and event-driven market responses
-   **Outcome**: Actionable investment strategies based on sentiment-driven market predictions

### Technology Stack

-   **Data Processing**: Python, Pandas, NumPy
-   **Financial Analysis**: TA-Lib, PyNance
-   **NLP & Sentiment**: NLTK, TextBlob
-   **Visualization**: Matplotlib, Seaborn
-   **Development**: Jupyter, pytest, GitHub Actions

## 🚀 Quick Start

### Prerequisites

-   **Python 3.8+** (Recommended: Python 3.11)
-   **Git** for version control
-   **4GB+ RAM** for processing large datasets

### Installation

1. **Clone and Setup:**

    ```bash
    git clone https://github.com/estif0/news-sentiment-price-prediction.git
    cd news-sentiment-price-prediction
    ```

2. **Create Virtual Environment:**

    ```bash
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Verify Installation:**

    ```bash
    # Run tests to verify setup
    pytest tests/ -v

    # Start Jupyter for exploration
    jupyter notebook notebooks/
    ```

## 📁 Project Structure

```
news-sentiment-price-prediction/
├── 📊 data/                    # Datasets (gitignored)
│   ├── raw/                    # Original datasets (1.4M+ news articles)
│   └── cleaned/                # Processed stock data (AAPL, AMZN, GOOG, META, MSFT, NVDA)
├── 📓 notebooks/              # Jupyter analysis notebooks
│   ├── 01_eda_news.ipynb     # Comprehensive EDA (Task 1)
│   ├── 02_quantitative_analysis.ipynb  # Technical indicators and financial metrics (Task 2)
│   └── 03_correlation_analysis.ipynb   # Sentiment-price correlation analysis (Task 3)
├── 📋 reports/               # Analysis reports
│   ├── interim_report.md     # Task 1 & 2 summary
│   └── final_report.md      # Complete project findings and recommendations
├── 🔧 scripts/               # Utility and automation scripts
├── 💻 src/                   # Source code modules
│   └── core/                 # Core business logic
│       ├── data_loader.py    # Data loading and validation
│       ├── eda.py           # Exploratory data analysis
│       ├── financial_analyzer.py # Technical indicators and financial metrics
│       ├── sentiment_analyzer.py # NLP sentiment scoring with TextBlob
│       ├── data_processor.py     # Date alignment and dataset merging
│       └── visualizer.py    # Plotting and visualization
├── 🧪 tests/                # Unit and integration tests (60 tests passing)
├── ⚙️ .github/workflows/    # CI/CD automation
├── 📋 requirements.txt      # Python dependencies
└── 📖 README.md            # This file
```

> **💡 Each directory contains its own README with detailed explanations**

## 📈 Dataset Overview

### Financial News Dataset (FNSPID)

-   **Volume**: 1,407,328 news articles
-   **Time Range**: 2012-2020 (8+ years)
-   **Sources**: 1,034+ unique publishers
-   **Coverage**: Major financial events, earnings reports, analyst ratings
-   **Quality**: 100% complete data, no missing values

### Key Statistics

-   **Average Headline Length**: 73.12 characters
-   **Peak Publishing**: 10 AM EST (aligned with market open)
-   **Event Sensitivity**: 61 major market events detected (COVID-19, earnings seasons)
-   **Publisher Concentration**: Top 3 publishers contribute 42% of content

## 🎯 Project Progress

### ✅ Task 1: Exploratory Data Analysis (Complete)

-   [x] **Repository Setup**: Git workflows, CI/CD, testing framework
-   [x] **Data Loading**: Robust data loader with error handling
-   [x] **Text Analysis**: Headline statistics, keyword extraction, topic modeling
-   [x] **Publisher Analysis**: Source identification, content specialization patterns
-   [x] **Temporal Analysis**: Publication frequency, market hours alignment, event detection
-   [x] **Data Quality**: Completeness assessment, bias identification
-   [x] **Professional Documentation**: Comprehensive notebook with data-driven insights
-   [x] **Interim Report**: Summarized findings and insights from Task 1

### ✅ Task 2: Quantitative Analysis (Complete)

-   [x] **Stock Data Integration**: Load and validate OHLCV price data for 6 major stocks
-   [x] **Technical Indicators**: Calculate SMA, EMA, RSI, MACD, Bollinger Bands using TA-Lib
-   [x] **Financial Metrics**: Volatility, returns, Sharpe ratio, max drawdown using PyNance
-   [x] **Data Visualization**: Interactive stock charts, multi-stock comparison, correlation heatmaps
-   [x] **Risk Analysis**: Portfolio performance metrics and comparative analysis

### ✅ Task 3: Correlation Analysis (Complete)

-   [x] **Data Alignment**: Synchronized news and stock datasets by date with UTC-4 handling
-   [x] **Sentiment Analysis**: NLP processing with TextBlob for sentiment polarity scoring
-   [x] **Daily Aggregation**: 1,588 daily sentiment-return records across 4 stocks
-   [x] **Statistical Correlation**: Pearson correlation analysis (NVDA: r=0.0992, p=0.0008)
-   [x] **Lag Analysis**: T+1, T+2, T+3 lagged correlations to test predictive power
-   [x] **Strategy Development**: Investment recommendations based on sentiment signals
-   [x] **Final Report**: Comprehensive analysis with statistical findings and insights

## 🛠️ Development Workflow

### Architecture Principles

-   **Object-Oriented Design**: Modular, reusable classes with clear interfaces
-   **Type Safety**: Full Python type hints for better code quality
-   **Test-Driven**: Comprehensive unit tests with CI integration
-   **Documentation**: Detailed docstrings and markdown explanations

### Code Quality Standards

-   **Formatting**: Black code formatter (enforced)
-   **Linting**: Flake8 compliance for style consistency
-   **Testing**: pytest with automated CI checks
-   **Version Control**: Conventional commits with descriptive messages

### Branch Strategy

-   `main`: Stable, production-ready code
-   `task-1`: EDA implementation (**Merged**)
-   `task-2`: Quantitative analysis (**Merged**)
-   `task-3`: Correlation modeling (**Merged**)

## 🏃‍♂️ Usage Examples

### Load and Analyze Data

```python
from src.core.data_loader import DataLoader
from src.core.eda import EDAAnalyzer
from src.core.visualizer import Visualizer

# Load financial news dataset
loader = DataLoader()
df = loader.load_news_data('data/raw/raw_analyst_ratings.csv')

# Perform analysis
analyzer = EDAAnalyzer(df)
headline_lengths = analyzer.calculate_headline_lengths()
keywords = analyzer.extract_common_keywords(top_n=20)
publication_frequency = analyzer.analyze_publication_frequency()

# Create visualizations
viz = Visualizer()
viz.plot_headline_length_distribution(headline_lengths)
viz.plot_common_keywords(keywords)
viz.plot_publication_frequency(publication_frequency)
```

### Analyze Stock Performance and Sentiment Correlation

```python
from src.core.data_loader import DataLoader
from src.core.financial_analyzer import FinancialAnalyzer
from src.core.sentiment_analyzer import SentimentAnalyzer
from src.core.data_processor import DataProcessor
from src.core.visualizer import Visualizer

# Load data
loader = DataLoader()
news_df = loader.load_news_data('data/raw/raw_analyst_ratings.csv')
stock_data = loader.load_stock_data('data/cleaned/AAPL.csv')

# Calculate financial metrics
fin_analyzer = FinancialAnalyzer()
metrics = fin_analyzer.calculate_performance_metrics(stock_data)
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

# Analyze sentiment and correlations
processor = DataProcessor()
merged_df = processor.process_and_merge(news_df, stock_data, 'AAPL')

sent_analyzer = SentimentAnalyzer()
sentiment_returns = sent_analyzer.aggregate_daily_sentiment(merged_df)
correlation = sentiment_returns['sentiment'].corr(sentiment_returns['daily_return'])
print(f"Sentiment-Return Correlation: {correlation:.4f}")

# Visualize
viz = Visualizer()
viz.plot_stock_overview(stock_data, "AAPL")
viz.plot_sentiment_vs_returns(sentiment_returns, "AAPL")
```

### Run Complete Analysis Pipeline

```bash
# Execute comprehensive analyses
jupyter notebook notebooks/01_eda_news.ipynb                # Task 1: EDA
jupyter notebook notebooks/02_quantitative_analysis.ipynb   # Task 2: Technical indicators
jupyter notebook notebooks/03_correlation_analysis.ipynb    # Task 3: Sentiment-price correlation

# Run all 60 unit tests
pytest tests/ -v --cov=src/

# Check code quality
black src/ && flake8 src/
```

## 📊 Key Findings

### 🔍 **Data Intelligence (Task 1)**

-   **Professional Quality**: Dataset suitable for institutional-grade analysis
-   **Market Alignment**: Strong correlation with US trading hours and market events
-   **Event Sensitivity**: System effectively captures major financial disruptions
-   **Publisher Landscape**: Benzinga ecosystem dominates with specialized coverage

### 📈 **Financial Analysis (Task 2)**

-   **Technical Indicators**: Successfully implemented SMA, EMA, RSI, MACD, Bollinger Bands
-   **Stock Coverage**: 6 major technology stocks analyzed (AAPL, AMZN, GOOG, META, MSFT, NVDA)
-   **Risk Metrics**: Comprehensive Sharpe ratio, max drawdown, and volatility analysis
-   **Visualization**: Professional-grade plots for technical analysis and portfolio comparison

### ⚡ **Sentiment-Price Correlation (Task 3)**

-   **Statistical Significance**: NVDA shows statistically significant correlation (r=0.0992, p=0.0008)
-   **Dataset Scale**: 5,064 merged news-stock records, 1,588 daily aggregations
-   **Sentiment Distribution**: 28.65% positive, 61.87% neutral, 9.48% negative headlines
-   **Predictive Insight**: Same-day correlations strongest, supporting efficient market hypothesis
-   **Investment Strategy**: Sentiment signals most effective for high-volume stocks (NVDA, GOOG)

## 🤝 Contributing

### For 10 Academy Students

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request with detailed description

### Development Guidelines

-   Follow existing code structure and naming conventions
-   Add unit tests for new functionality (currently 60 tests passing)
-   Update documentation for any API changes
-   Ensure all CI checks pass before submitting PR
-   Use type hints and comprehensive docstrings

## 📋 Project Timeline

| Phase      | Deadline     | Status      | Deliverables                                           |
| ---------- | ------------ | ----------- | ------------------------------------------------------ |
| **Task 1** | Nov 23, 2025 | ✅ Complete | EDA notebook, interim report                           |
| **Task 2** | Nov 25, 2025 | ✅ Complete | Quantitative analysis, technical indicators (32 tests) |
| **Task 3** | Nov 26, 2025 | ✅ Complete | Sentiment correlation, final report (60 tests total)   |

**🎉 Project Complete**: All three tasks delivered with comprehensive analysis, statistical validation, and actionable investment insights.

## 📞 Support & Resources

-   **Code Examples**: `/notebooks/` directory (3 comprehensive notebooks)
-   **API Reference**: Docstrings in `/src/core/` modules (6 classes)
-   **Analysis Reports**: `/reports/` directory (interim and final reports)
-   **Issue Tracking**: GitHub Issues for bugs and feature requests

## 📄 License

This project is part of the 10 Academy AI Mastery program. See LICENSE file for details.

---

**✅ Project Successfully Completed - Ready for Submission!**

**Key Achievements:**

-   🎯 All 3 tasks completed with professional-grade analysis
-   📊 60 unit tests passing (100% success rate)
-   📈 Statistically significant sentiment-price correlation discovered
-   📋 Comprehensive final report with actionable investment strategies
