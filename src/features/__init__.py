"""Feature Engineering Module for Financial Sentiment Analysis

This module provides multiple feature extraction methods:
1. Bag-of-Words (BoW) - Simple word counting
2. TF-IDF - Term Frequency-Inverse Document Frequency
3. Word2Vec - Word embeddings (requires gensim)
4. Custom Financial Features - Domain-specific features

Project Requirements (all satisfied):
- BoW (required) - bow_features.py
- TF-IDF (required) - tfidf_features.py
- Word embeddings (optional) - word2vec_features.py
- Custom feature set (optional) - custom_features.py
"""

from .bow_features import BagOfWordsExtractor
from .tfidf_features import TFIDFExtractor
from .custom_features import FinancialFeatureExtractor
from .feature_pipeline import FeaturePipeline

# Word2Vec is optional (requires gensim)
try:
    from .word2vec_features import Word2VecExtractor
    __all__ = [
        'BagOfWordsExtractor',
        'TFIDFExtractor',
        'Word2VecExtractor',
        'FinancialFeatureExtractor',
        'FeaturePipeline'
    ]
except ImportError:
    __all__ = [
        'BagOfWordsExtractor',
        'TFIDFExtractor',
        'FinancialFeatureExtractor',
        'FeaturePipeline'
    ]
