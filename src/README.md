# Source Code Directory

This directory contains the main source code modules for the News Sentiment Price Prediction project.

## Structure

### `core/` - Core Business Logic

Contains the main analysis classes implementing the project's core functionality:

-   **`data_loader.py`**: Data loading and validation for news and stock datasets
-   **`eda.py`**: Exploratory Data Analysis with NLTK integration
-   **`financial_analyzer.py`**: Technical indicators (TA-Lib) and financial metrics (PyNance)
-   **`sentiment_analyzer.py`**: NLP sentiment scoring with TextBlob polarity analysis
-   **`data_processor.py`**: Date alignment and news-stock dataset merging
-   **`visualizer.py`**: Professional plotting and visualization functions

## Design Principles

### Object-Oriented Architecture

-   **Modular Classes**: Each module contains focused, single-responsibility classes
-   **Cooperative Design**: Classes work together seamlessly (e.g., `DataLoader` → `EDAAnalyzer` → `Visualizer`)
-   **Extensible Framework**: Easy to add new analyzers for Tasks 2 & 3

### Code Quality Standards

-   **Type Hints**: Full Python typing for all methods and functions
-   **Documentation**: Comprehensive docstrings for all classes and methods
-   **Error Handling**: Robust exception handling with informative messages
-   **Testing**: Unit tests for all core functionality

## Usage Examples

### Basic EDA Workflow

```python
from src.core.data_loader import DataLoader
from src.core.eda import EDAAnalyzer
from src.core.visualizer import Visualizer

# Load data
loader = DataLoader()
df = loader.load_news_data('data/raw/news_data.csv')

# Analyze
analyzer = EDAAnalyzer(df)
headline_lengths = analyzer.calculate_headline_lengths()
keywords = analyzer.extract_common_keywords(top_n=20)

# Visualize
viz = Visualizer()
viz.plot_headline_length_distribution(headline_lengths)
viz.plot_common_keywords(keywords)
```

### Financial Analysis Workflow

```python
from src.core.data_loader import DataLoader
from src.core.financial_analyzer import FinancialAnalyzer
from src.core.visualizer import Visualizer

# Load stock data
loader = DataLoader()
stock_df = loader.load_stock_data('data/cleaned/AAPL.csv')

# Calculate technical indicators
analyzer = FinancialAnalyzer()
stock_with_indicators = analyzer.add_technical_indicators(stock_df)
metrics = analyzer.calculate_performance_metrics(stock_df)

# Visualize
viz = Visualizer()
viz.plot_technical_dashboard(stock_with_indicators, 'AAPL')
```

### Sentiment Correlation Workflow

```python
from src.core.data_loader import DataLoader
from src.core.data_processor import DataProcessor
from src.core.sentiment_analyzer import SentimentAnalyzer

# Load datasets
loader = DataLoader()
news_df = loader.load_news_data('data/raw/raw_analyst_ratings.csv')
stock_df = loader.load_stock_data('data/cleaned/NVDA.csv')

# Process and merge
processor = DataProcessor()
merged_df = processor.process_and_merge(news_df, stock_df, 'NVDA')

# Analyze sentiment
sent_analyzer = SentimentAnalyzer()
sentiment_returns = sent_analyzer.aggregate_daily_sentiment(merged_df)
correlation = sentiment_returns['sentiment'].corr(sentiment_returns['daily_return'])
print(f"Correlation: {correlation:.4f}")
```

## Planned Expansions

### Future Enhancements

-   **Machine Learning Models**: LSTM/Transformer models for sentiment-based price prediction
-   **Real-time Processing**: Live news feed integration for real-time sentiment monitoring
-   **Advanced NLP**: Fine-tuned FinBERT for domain-specific financial sentiment
-   **Portfolio Optimization**: Multi-stock portfolio construction using sentiment signals
-   **Risk Management**: VaR and CVaR calculations incorporating sentiment factors

## Development Guidelines

-   Follow PEP 8 style conventions (enforced by Black formatter)
-   Add type hints for all new functions and methods
-   Write comprehensive docstrings with parameters and return types
-   Include error handling for edge cases and invalid inputs
-   Write unit tests for all new functionality
