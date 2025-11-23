# Notebooks Directory

This directory contains Jupyter notebooks for interactive data exploration, analysis, and reporting.

## Contents

### `01_eda_news.ipynb`

-   **Purpose**: Comprehensive Exploratory Data Analysis (EDA) of the financial news dataset
-   **Dataset**: 1.4M+ financial news articles (2012-2020)
-   **Analysis Includes**:
    -   Text characteristics and headline analysis
    -   Publisher landscape and content specialization
    -   Temporal patterns and publication frequency
    -   Event detection and market spike analysis
    -   Data quality assessment and insights

## Usage

```bash
# Start Jupyter from project root
jupyter notebook notebooks/

# Or open specific notebook
jupyter notebook notebooks/01_eda_news.ipynb
```

## Dependencies

All notebooks use the custom analysis modules from `src/core/`:

-   `DataLoader`: Load and validate datasets
-   `EDAAnalyzer`: Perform statistical and textual analysis
-   `Visualizer`: Generate plots and charts

## Notes

-   Notebooks are designed to run from the `notebooks/` directory
-   Auto-reload is enabled for dynamic module development
-   All outputs and visualizations are preserved for documentation
