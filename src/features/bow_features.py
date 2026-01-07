"""Bag-of-Words Feature Extraction

Simple word counting vectorization for baseline comparison.
Required by project specifications.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Tuple, Optional
import pickle


class BagOfWordsExtractor:
    """
    Bag-of-Words feature extractor for financial text.

    Creates sparse word count vectors from text documents.
    Used as baseline for comparison with TF-IDF and embeddings.

    Parameters:
    -----------
    max_features : int
        Maximum number of features (vocabulary size)
    ngram_range : tuple
        Range of n-grams to extract (default: unigrams only)
    min_df : int or float
        Minimum document frequency for terms
    max_df : float
        Maximum document frequency for terms
    """

    def __init__(
        self,
        max_features: int = 1000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95
    ):
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        self.is_fitted = False

    def fit(self, texts: List[str]) -> 'BagOfWordsExtractor':
        """
        Fit the BoW vectorizer on training texts.

        Parameters:
        -----------
        texts : List[str]
            Training documents

        Returns:
        --------
        self
        """
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to BoW vectors.

        Parameters:
        -----------
        texts : List[str]
            Documents to transform

        Returns:
        --------
        np.ndarray
            Dense BoW feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        return self.vectorizer.transform(texts).toarray()

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self) -> List[str]:
        """Get vocabulary terms."""
        return list(self.vectorizer.get_feature_names_out())

    def get_vocabulary_size(self) -> int:
        """Get number of features."""
        return len(self.vectorizer.vocabulary_)

    def save(self, filepath: str):
        """Save vectorizer to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load(self, filepath: str):
        """Load vectorizer from file."""
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True


def extract_bow_features(
    train_texts: List[str],
    test_texts: List[str],
    val_texts: Optional[List[str]] = None,
    max_features: int = 1000,
    ngram_range: Tuple[int, int] = (1, 2)
) -> dict:
    """
    Extract BoW features for train/test/val splits.

    Parameters:
    -----------
    train_texts : List[str]
        Training documents
    test_texts : List[str]
        Test documents
    val_texts : List[str], optional
        Validation documents
    max_features : int
        Maximum vocabulary size
    ngram_range : tuple
        N-gram range

    Returns:
    --------
    dict with keys: 'train', 'test', 'val', 'vectorizer', 'feature_names'
    """
    extractor = BagOfWordsExtractor(
        max_features=max_features,
        ngram_range=ngram_range
    )

    # Fit on training data
    X_train = extractor.fit_transform(train_texts)
    X_test = extractor.transform(test_texts)

    result = {
        'train': X_train,
        'test': X_test,
        'vectorizer': extractor,
        'feature_names': extractor.get_feature_names(),
        'n_features': extractor.get_vocabulary_size()
    }

    if val_texts is not None:
        result['val'] = extractor.transform(val_texts)

    return result


if __name__ == "__main__":
    # Test BoW extractor
    sample_texts = [
        "Stock prices surged after earnings report",
        "Market crashed due to economic concerns",
        "Company maintains steady growth outlook"
    ]

    extractor = BagOfWordsExtractor(max_features=100)
    features = extractor.fit_transform(sample_texts)

    print(f"BoW Features shape: {features.shape}")
    print(f"Vocabulary size: {extractor.get_vocabulary_size()}")
    print(f"Sample features: {extractor.get_feature_names()[:10]}")
