"""Custom Financial Feature Extraction

Domain-specific features for financial sentiment analysis.
Captures linguistic patterns specific to financial text.
"""

import numpy as np
import re
from typing import List, Dict
from collections import Counter


class FinancialFeatureExtractor:
    """
    Custom feature extractor for financial text.

    Extracts domain-specific features that may not be captured
    by bag-of-words or TF-IDF approaches:

    1. Sentiment word counts (positive/negative keywords)
    2. Numeric features (percentages, dollar amounts)
    3. Financial entity mentions (company types, stock symbols)
    4. Text statistics (length, complexity)
    5. Punctuation patterns (exclamation marks, question marks)

    These features can be combined with BoW/TF-IDF for improved performance.
    """

    def __init__(self):
        # Positive financial keywords
        self.positive_words = {
            'surge', 'surged', 'soar', 'soared', 'rally', 'rallied',
            'gain', 'gains', 'profit', 'profits', 'growth', 'growing',
            'bullish', 'optimistic', 'upgrade', 'beat', 'beats',
            'exceeded', 'strong', 'positive', 'rise', 'rising',
            'high', 'higher', 'outperform', 'success', 'boom'
        }

        # Negative financial keywords
        self.negative_words = {
            'crash', 'crashed', 'plunge', 'plunged', 'collapse', 'collapsed',
            'loss', 'losses', 'decline', 'declining', 'bearish', 'pessimistic',
            'downgrade', 'miss', 'missed', 'weak', 'weakness', 'fear',
            'drop', 'dropped', 'fall', 'falling', 'low', 'lower',
            'underperform', 'failure', 'bust', 'deficit', 'layoff'
        }

        # Neutral financial keywords
        self.neutral_words = {
            'unchanged', 'steady', 'stable', 'flat', 'mixed',
            'hold', 'maintain', 'neutral', 'sideways', 'consolidate',
            'wait', 'expect', 'announce', 'report', 'update'
        }

        # Financial entities
        self.company_types = {
            'corp', 'corporation', 'inc', 'company', 'ltd', 'limited',
            'group', 'holdings', 'industries', 'enterprise', 'bank'
        }

        # Regex patterns
        self.percentage_pattern = re.compile(r'\d+\.?\d*%')
        self.dollar_pattern = re.compile(r'\$\d+\.?\d*[BMK]?')
        self.ticker_pattern = re.compile(r'\$[A-Z]{1,5}\b')
        self.number_pattern = re.compile(r'\b\d+\.?\d*\b')

    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract all custom features from a single text.

        Parameters:
        -----------
        text : str
            Input financial text

        Returns:
        --------
        dict
            Feature name -> value mapping
        """
        text_lower = text.lower()
        words = text_lower.split()

        features = {}

        # 1. Sentiment word counts
        features['positive_word_count'] = sum(1 for w in words if w in self.positive_words)
        features['negative_word_count'] = sum(1 for w in words if w in self.negative_words)
        features['neutral_word_count'] = sum(1 for w in words if w in self.neutral_words)

        # Sentiment ratios
        total_sentiment = (features['positive_word_count'] +
                          features['negative_word_count'] +
                          features['neutral_word_count'])
        if total_sentiment > 0:
            features['positive_ratio'] = features['positive_word_count'] / total_sentiment
            features['negative_ratio'] = features['negative_word_count'] / total_sentiment
        else:
            features['positive_ratio'] = 0.0
            features['negative_ratio'] = 0.0

        # 2. Numeric features
        features['percentage_count'] = len(self.percentage_pattern.findall(text))
        features['dollar_count'] = len(self.dollar_pattern.findall(text))
        features['number_count'] = len(self.number_pattern.findall(text))
        features['ticker_count'] = len(self.ticker_pattern.findall(text))

        # 3. Entity features
        features['company_mention'] = sum(1 for w in words if w in self.company_types)

        # 4. Text statistics
        features['word_count'] = len(words)
        features['char_count'] = len(text)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0

        # Vocabulary richness (unique words / total words)
        features['vocabulary_richness'] = len(set(words)) / len(words) if words else 0

        # 5. Punctuation patterns
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0

        return features

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform list of texts to feature matrix.

        Parameters:
        -----------
        texts : List[str]
            Input documents

        Returns:
        --------
        np.ndarray
            Feature matrix (n_samples, n_features)
        """
        all_features = [self.extract_features(text) for text in texts]

        # Get consistent feature ordering
        feature_names = sorted(all_features[0].keys())

        # Create matrix
        matrix = np.array([
            [features[name] for name in feature_names]
            for features in all_features
        ])

        return matrix

    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names."""
        dummy_features = self.extract_features("dummy text")
        return sorted(dummy_features.keys())

    def get_n_features(self) -> int:
        """Get number of features."""
        return len(self.get_feature_names())


def extract_custom_features(
    train_texts: List[str],
    test_texts: List[str],
    val_texts: List[str] = None
) -> dict:
    """
    Extract custom features for all splits.

    Parameters:
    -----------
    train_texts : List[str]
        Training documents
    test_texts : List[str]
        Test documents
    val_texts : List[str], optional
        Validation documents

    Returns:
    --------
    dict with keys: 'train', 'test', 'val', 'extractor', 'feature_names'
    """
    extractor = FinancialFeatureExtractor()

    result = {
        'train': extractor.transform(train_texts),
        'test': extractor.transform(test_texts),
        'extractor': extractor,
        'feature_names': extractor.get_feature_names(),
        'n_features': extractor.get_n_features()
    }

    if val_texts is not None:
        result['val'] = extractor.transform(val_texts)

    return result


if __name__ == "__main__":
    # Test custom feature extractor
    sample_texts = [
        "Stock prices surged 15% after strong earnings report!",
        "$AAPL crashed 20% due to weak iPhone sales",
        "Market remained steady with mixed signals"
    ]

    extractor = FinancialFeatureExtractor()

    print("Custom Financial Features:\n")
    for text in sample_texts:
        features = extractor.extract_features(text)
        print(f"Text: {text}")
        print(f"  Positive words: {features['positive_word_count']}")
        print(f"  Negative words: {features['negative_word_count']}")
        print(f"  Percentages: {features['percentage_count']}")
        print(f"  Tickers: {features['ticker_count']}")
        print()

    # Transform to matrix
    matrix = extractor.transform(sample_texts)
    print(f"Feature matrix shape: {matrix.shape}")
    print(f"Feature names: {extractor.get_feature_names()}")
