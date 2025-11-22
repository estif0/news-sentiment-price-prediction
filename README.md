# News Sentiment Price Prediction

This project analyzes financial news sentiment to predict stock price movements. It involves Data Engineering, Financial Analytics, and Machine Learning.

## Project Overview

-   **Goal**: Correlate news sentiment (headline text) with stock price movements.
-   **Stack**: Python, Pandas, TA-Lib, PyNance, NLTK/TextBlob, Scikit-learn.

## Setup Instructions

### Prerequisites

-   Python 3.8+
-   Git

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/estif0/news-sentiment-price-prediction.git
    cd news-sentiment-price-prediction
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

-   `.github/`: CI/CD configurations.
-   `.vscode/`: VS Code settings.
-   `data/`: Data storage (gitignored).
    -   `raw/`: Raw datasets.
    -   `cleaned/`: Processed datasets.
-   `docs/`: Documentation.
-   `notebooks/`: Jupyter notebooks for exploration and reporting.
-   `scripts/`: Utility scripts.
-   `src/`: Source code modules.
    -   `core/`: Core logic modules.
-   `tests/`: Unit tests.
