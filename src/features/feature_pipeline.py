"""Feature Engineering Pipeline

Combines all feature extraction methods into a unified pipeline.
Supports multiple feature types: BoW, TF-IDF, Custom, Combined.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import pickle

from .bow_features import BagOfWordsExtractor
from .tfidf_features import TFIDFExtractor
from .custom_features import FinancialFeatureExtractor


class FeaturePipeline:
    """
    Unified feature extraction pipeline.

    Combines multiple feature extraction methods:
    1. Bag-of-Words (BoW)
    2. TF-IDF
    3. Custom financial features
    4. Combined (concatenation)

    Parameters:
    -----------
    feature_type : str
        Type of features to extract:
        - 'bow': Bag-of-Words only
        - 'tfidf': TF-IDF only (default)
        - 'custom': Custom features only
        - 'tfidf+custom': TF-IDF + Custom combined
        - 'bow+custom': BoW + Custom combined
        - 'all': All features combined

    max_features : int
        Maximum features for BoW/TF-IDF (default: 1000)

    ngram_range : tuple
        N-gram range for BoW/TF-IDF (default: (1, 3))
    """

    def __init__(
        self,
        feature_type: str = 'tfidf',
        max_features: int = 1000,
        ngram_range: Tuple[int, int] = (1, 3)
    ):
        self.feature_type = feature_type
        self.max_features = max_features
        self.ngram_range = ngram_range

        # Initialize extractors based on type
        self.bow_extractor = None
        self.tfidf_extractor = None
        self.custom_extractor = None

        if feature_type in ['bow', 'bow+custom', 'all']:
            self.bow_extractor = BagOfWordsExtractor(
                max_features=max_features,
                ngram_range=ngram_range
            )

        if feature_type in ['tfidf', 'tfidf+custom', 'all']:
            self.tfidf_extractor = TFIDFExtractor(
                max_features=max_features,
                ngram_range=ngram_range
            )

        if feature_type in ['custom', 'tfidf+custom', 'bow+custom', 'all']:
            self.custom_extractor = FinancialFeatureExtractor()

        self.is_fitted = False
        self.feature_info = {}

    def fit(self, texts: List[str]) -> 'FeaturePipeline':
        """
        Fit all extractors on training texts.

        Parameters:
        -----------
        texts : List[str]
            Training documents

        Returns:
        --------
        self
        """
        if self.bow_extractor is not None:
            self.bow_extractor.fit(texts)

        if self.tfidf_extractor is not None:
            self.tfidf_extractor.fit(texts)

        # Custom extractor doesn't need fitting

        self.is_fitted = True
        self._update_feature_info()
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to feature matrix.

        Parameters:
        -----------
        texts : List[str]
            Documents to transform

        Returns:
        --------
        np.ndarray
            Combined feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        features = []

        if self.bow_extractor is not None:
            features.append(self.bow_extractor.transform(texts))

        if self.tfidf_extractor is not None:
            features.append(self.tfidf_extractor.transform(texts))

        if self.custom_extractor is not None:
            features.append(self.custom_extractor.transform(texts))

        # Concatenate all features
        if len(features) == 1:
            return features[0]
        else:
            return np.hstack(features)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)

    def _update_feature_info(self):
        """Update feature information."""
        self.feature_info = {
            'type': self.feature_type,
            'total_features': 0,
            'components': []
        }

        if self.bow_extractor is not None and self.bow_extractor.is_fitted:
            n = self.bow_extractor.get_vocabulary_size()
            self.feature_info['bow_features'] = n
            self.feature_info['total_features'] += n
            self.feature_info['components'].append(f'BoW ({n})')

        if self.tfidf_extractor is not None and self.tfidf_extractor.is_fitted:
            n = self.tfidf_extractor.get_vocabulary_size()
            self.feature_info['tfidf_features'] = n
            self.feature_info['total_features'] += n
            self.feature_info['components'].append(f'TF-IDF ({n})')

        if self.custom_extractor is not None:
            n = self.custom_extractor.get_n_features()
            self.feature_info['custom_features'] = n
            self.feature_info['total_features'] += n
            self.feature_info['components'].append(f'Custom ({n})')

    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about extracted features."""
        return self.feature_info

    def get_n_features(self) -> int:
        """Get total number of features."""
        return self.feature_info.get('total_features', 0)

    def save(self, filepath: str):
        """Save pipeline to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'feature_type': self.feature_type,
                'max_features': self.max_features,
                'ngram_range': self.ngram_range,
                'bow_extractor': self.bow_extractor,
                'tfidf_extractor': self.tfidf_extractor,
                'custom_extractor': self.custom_extractor,
                'feature_info': self.feature_info
            }, f)

    def load(self, filepath: str):
        """Load pipeline from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.feature_type = data['feature_type']
            self.max_features = data['max_features']
            self.ngram_range = data['ngram_range']
            self.bow_extractor = data['bow_extractor']
            self.tfidf_extractor = data['tfidf_extractor']
            self.custom_extractor = data['custom_extractor']
            self.feature_info = data['feature_info']
            self.is_fitted = True


def compare_feature_methods(
    train_texts: List[str],
    test_texts: List[str],
    train_labels: np.ndarray,
    test_labels: np.ndarray
) -> Dict[str, Dict]:
    """
    Compare different feature extraction methods.

    Returns performance comparison for:
    - BoW
    - TF-IDF
    - Custom
    - TF-IDF + Custom

    Parameters:
    -----------
    train_texts, test_texts : List[str]
        Train and test documents
    train_labels, test_labels : np.ndarray
        Train and test labels

    Returns:
    --------
    dict with results for each method
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    results = {}
    methods = ['bow', 'tfidf', 'custom', 'tfidf+custom']

    for method in methods:
        pipeline = FeaturePipeline(feature_type=method)

        X_train = pipeline.fit_transform(train_texts)
        X_test = pipeline.transform(test_texts)

        # Quick evaluation with LogReg
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, train_labels)
        y_pred = model.predict(X_test)

        f1 = f1_score(test_labels, y_pred, average='macro')

        results[method] = {
            'n_features': pipeline.get_n_features(),
            'f1_score': f1,
            'feature_info': pipeline.get_feature_info()
        }

    return results


if __name__ == "__main__":
    # Test pipeline
    sample_texts = [
        "Stock prices surged 15% after strong earnings report",
        "Market crashed due to weak economic data",
        "Company maintains steady outlook for Q4"
    ]

    print("Testing Feature Pipeline:\n")

    for feature_type in ['bow', 'tfidf', 'custom', 'tfidf+custom']:
        pipeline = FeaturePipeline(feature_type=feature_type)
        features = pipeline.fit_transform(sample_texts)

        info = pipeline.get_feature_info()
        print(f"{feature_type.upper()}:")
        print(f"  Shape: {features.shape}")
        print(f"  Components: {info['components']}")
        print()
