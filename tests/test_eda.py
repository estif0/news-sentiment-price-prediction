import unittest
import pandas as pd
from src.core.eda import EDAAnalyzer

class TestEDAAnalyzer(unittest.TestCase):
    def setUp(self):
        data = {
            'headline': ['Stock goes up', 'Market crash imminent', 'Buy now'],
            'publisher': ['Pub A', 'Pub B', 'Pub A']
        }
        self.df = pd.DataFrame(data)
        self.analyzer = EDAAnalyzer(self.df)

    def test_calculate_headline_lengths(self):
        lengths = self.analyzer.calculate_headline_lengths()
        # 'Stock goes up' -> 13
        # 'Market crash imminent' -> 21
        # 'Buy now' -> 7
        self.assertEqual(lengths.tolist(), [13, 21, 7])

    def test_count_articles_per_publisher(self):
        counts = self.analyzer.count_articles_per_publisher()
        self.assertEqual(counts['Pub A'], 2)
        self.assertEqual(counts['Pub B'], 1)

    def test_extract_common_keywords(self):
        # "stock goes up market crash imminent buy now"
        # Stopwords removed: stock, goes, market, crash, imminent, buy
        # 'now' might be a stopword depending on NLTK list, usually it is.
        # 'up' is a stopword.
        
        # Let's use a simpler deterministic case
        data = {'headline': ['apple apple banana', 'apple banana orange']}
        df = pd.DataFrame(data)
        analyzer = EDAAnalyzer(df)
        
        keywords = analyzer.extract_common_keywords(top_n=3)
        # apple: 3, banana: 2, orange: 1
        
        expected = [('apple', 3), ('banana', 2), ('orange', 1)]
        self.assertEqual(keywords, expected)

if __name__ == '__main__':
    unittest.main()
