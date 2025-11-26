# Tests Directory

This directory contains unit tests and integration tests for the project's core functionality.

## Test Structure

### Unit Tests

-   `test_data_loader.py`: Tests for data loading and validation (Task 1)
-   `test_eda.py`: Tests for exploratory data analysis functions (Task 1)
-   `test_financial_analyzer.py`: Tests for technical indicators and financial metrics (Task 2)
-   `test_data_processor.py`: Tests for date alignment and dataset merging (Task 3)
-   `test_sentiment_analyzer.py`: Tests for sentiment scoring and aggregation (Task 3)
-   `test_stock_loader.py`: Tests for stock data loading functionality
-   `test_smoke.py`: Basic smoke tests for project setup

### Test Coverage

| Module              | Tests | Status                                       |
| ------------------- | ----- | -------------------------------------------- |
| `DataLoader`        | ✅    | Loading, error handling, validation          |
| `EDAAnalyzer`       | ✅    | Text analysis, statistics, caching           |
| `FinancialAnalyzer` | ✅    | Technical indicators, risk metrics (32)      |
| `DataProcessor`     | ✅    | Date alignment, merging, timezone handling   |
| `SentimentAnalyzer` | ✅    | Sentiment scoring, aggregation, distribution |
| `Visualizer`        | 🚧    | Plotting tested manually via notebooks       |

**Total: 60 tests passing** (100% success rate)

## Running Tests

### All Tests

```bash
# From project root
pytest tests/

# With verbose output
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src/
```

### Specific Tests

```bash
# Single test file
pytest tests/test_data_loader.py

# Specific test method
pytest tests/test_eda.py::TestEDAAnalyzer::test_calculate_headline_lengths
```

## Test Guidelines

### Writing Tests

-   Follow `test_*.py` naming convention
-   Use descriptive test method names
-   Include both positive and negative test cases
-   Mock external dependencies (files, APIs, etc.)

### Test Data

-   Create minimal test datasets in `setUp()` methods
-   Clean up temporary files in `tearDown()` methods
-   Use fixtures for reusable test data

## CI Integration

Tests are automatically run via GitHub Actions:

-   **Trigger**: Every push and pull request
-   **Python Version**: 3.11
-   **Dependencies**: Auto-installed from `requirements.txt`
-   **Coverage**: Reports generated for code quality monitoring

## Current Status

✅ **60/60 tests passing** (100% success rate)  
✅ **CI pipeline configured and passing**  
✅ **Code quality checks enabled**  
✅ **All three tasks validated**

### Test Statistics by Task

-   **Task 1 (EDA)**: ~10 tests
-   **Task 2 (Quantitative)**: 32 tests (technical indicators + financial metrics)
-   **Task 3 (Correlation)**: 29 tests (10 DataProcessor + 19 SentimentAnalyzer)
-   **Infrastructure**: ~5 smoke and integration tests
