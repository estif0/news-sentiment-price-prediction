import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Tuple

class Visualizer:
    """
    Class for visualizing financial news data and analysis results.
    """

    def __init__(self):
        """
        Initialize the Visualizer.
        Sets the style for seaborn plots.
        """
        sns.set_theme(style="whitegrid")

    def plot_headline_length_distribution(self, lengths: pd.Series, bins: int = 30):
        """
        Plots the distribution of headline lengths.

        Args:
            lengths (pd.Series): Series containing headline lengths.
            bins (int): Number of bins for the histogram.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(lengths, bins=bins, kde=True)
        plt.title('Distribution of Headline Lengths')
        plt.xlabel('Length (characters)')
        plt.ylabel('Frequency')
        plt.show()

    def plot_top_publishers(self, publisher_counts: pd.Series, top_n: int = 10):
        """
        Plots the top N publishers by article count.

        Args:
            publisher_counts (pd.Series): Series with publisher names as index and counts as values.
            top_n (int): Number of top publishers to plot.
        """
        top_publishers = publisher_counts.head(top_n)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_publishers.values, y=top_publishers.index, hue=top_publishers.index, palette="viridis", legend=False)
        plt.title(f'Top {top_n} Publishers by Article Count')
        plt.xlabel('Number of Articles')
        plt.ylabel('Publisher')
        plt.show()

    def plot_common_keywords(self, keywords: List[Tuple[str, int]]):
        """
        Plots the most common keywords.

        Args:
            keywords (List[Tuple[str, int]]): List of tuples (keyword, count).
        """
        words, counts = zip(*keywords)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(counts), y=list(words), hue=list(words), palette="magma", legend=False)
        plt.title('Top Common Keywords in Headlines')
        plt.xlabel('Frequency')
        plt.ylabel('Keyword')
        plt.show()

    def plot_publication_frequency(self, frequency: pd.Series, title: str = 'Article Publication Frequency'):
        """
        Plots the publication frequency over time.

        Args:
            frequency (pd.Series): Series with date index and count values.
            title (str): Title of the plot.
        """
        plt.figure(figsize=(14, 7))
        frequency.plot()
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.show()

    def plot_publishing_times(self, hourly_counts: pd.Series):
        """
        Plots the distribution of publishing times (by hour).

        Args:
            hourly_counts (pd.Series): Series with hour index and count values.
        """
        plt.figure(figsize=(12, 6))
        sns.barplot(x=hourly_counts.index, y=hourly_counts.values, hue=hourly_counts.index, palette="coolwarm", legend=False)
        plt.title('Article Publication by Hour of Day (UTC)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Articles')
        plt.show()
