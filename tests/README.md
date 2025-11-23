# Tests Directory

This directory contains unit tests and integration tests for the project's core functionality.

## Test Structure

### Unit Tests

-   `test_data_loader.py`: Tests for data loading and validation
-   `test_eda.py`: Tests for exploratory data analysis functions
-   `test_smoke.py`: Basic smoke tests for project setup

### Test Coverage

| Module              | Tests | Status                    |
| ------------------- | ----- | ------------------------- |
| `DataLoader`        | ✅    | Loading, error handling   |
| `EDAAnalyzer`       | ✅    | Text analysis, statistics |
| `Visualizer`        | 🚧    | Planned                   |
| `SentimentAnalyzer` | 🚧    | Task 3                    |
| `FinancialAnalyzer` | 🚧    | Task 2                    |

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

✅ **6/6 tests passing**  
✅ **CI pipeline configured**  
✅ **Code quality checks enabled**
