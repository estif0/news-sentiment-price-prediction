# Source Code Directory

This directory contains the main source code modules for the News Sentiment Price Prediction project.

## Structure

### `core/` - Core Business Logic

Contains the main analysis classes implementing the project's core functionality:

-   **`data_loader.py`**: Data loading and validation
-   **`eda.py`**: Exploratory Data Analysis with NLTK integration
-   **`visualizer.py`**: Plotting and visualization functions

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

## Planned Expansions

### Task 2: Quantitative Analysis

-   `financial_analyzer.py`: Technical indicators, stock metrics
-   `stock_data_loader.py`: Stock price data management
-   `indicator_calculator.py`: TA-Lib integration

### Task 3: Correlation Analysis

-   `sentiment_analyzer.py`: NLP sentiment scoring
-   `correlation_engine.py`: Statistical correlation analysis
-   `data_processor.py`: Date alignment and data merging

## Development Guidelines

-   Follow PEP 8 style conventions (enforced by Black formatter)
-   Add type hints for all new functions and methods
-   Write comprehensive docstrings with parameters and return types
-   Include error handling for edge cases and invalid inputs
-   Write unit tests for all new functionality
