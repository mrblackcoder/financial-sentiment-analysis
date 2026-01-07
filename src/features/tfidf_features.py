"""TF-IDF Feature Extraction

Term Frequency-Inverse Document Frequency vectorization.
Primary feature extraction method for this project.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Optional
import pickle


class TFIDFExtractor:
    """
    TF-IDF feature extractor for financial text.

    TF-IDF captures word importance by considering both:
    - Term Frequency (TF): How often a word appears in a document
    - Inverse Document Frequency (IDF): How rare a word is across all documents

    This is particularly effective for financial text where
    keywords like 'profit', 'loss', 'surge' directly indicate sentiment.

    Parameters:
    -----------
    max_features : int
        Maximum number of features (default: 1000)
    ngram_range : tuple
        Range of n-grams (default: 1-3 for unigrams, bigrams, trigrams)
    min_df : int or float
        Minimum document frequency
    max_df : float
        Maximum document frequency
    sublinear_tf : bool
        Apply sublinear tf scaling (log(1 + tf))
    """

    def __init__(
        self,
        max_features: int = 1000,
        ngram_range: Tuple[int, int] = (1, 3),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            sublinear_tf=sublinear_tf
        )
        self.is_fitted = False

    def fit(self, texts: List[str]) -> 'TFIDFExtractor':
        """
        Fit the TF-IDF vectorizer on training texts.

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
        Transform texts to TF-IDF vectors.

        Parameters:
        -----------
        texts : List[str]
            Documents to transform

        Returns:
        --------
        np.ndarray
            Dense TF-IDF feature matrix
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

    def get_idf_scores(self) -> dict:
        """Get IDF scores for all terms."""
        feature_names = self.get_feature_names()
        return dict(zip(feature_names, self.vectorizer.idf_))

    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """Get top N features by IDF score."""
        idf_scores = self.get_idf_scores()
        sorted_features = sorted(idf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]

    def save(self, filepath: str):
        """Save vectorizer to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load(self, filepath: str):
        """Load vectorizer from file."""
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True


def extract_tfidf_features(
    train_texts: List[str],
    test_texts: List[str],
    val_texts: Optional[List[str]] = None,
    max_features: int = 1000,
    ngram_range: Tuple[int, int] = (1, 3)
) -> dict:
    """
    Extract TF-IDF features for train/test/val splits.

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
    extractor = TFIDFExtractor(
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
        'n_features': extractor.get_vocabulary_size(),
        'top_features': extractor.get_top_features(20)
    }

    if val_texts is not None:
        result['val'] = extractor.transform(val_texts)

    return result


if __name__ == "__main__":
    # Test TF-IDF extractor
    sample_texts = [
        "Stock prices surged after strong earnings report",
        "Market crashed due to economic concerns and losses",
        "Company maintains steady growth outlook for quarter"
    ]

    extractor = TFIDFExtractor(max_features=100)
    features = extractor.fit_transform(sample_texts)

    print(f"TF-IDF Features shape: {features.shape}")
    print(f"Vocabulary size: {extractor.get_vocabulary_size()}")
    print(f"\nTop 10 features by IDF:")
    for term, score in extractor.get_top_features(10):
        print(f"  {term}: {score:.4f}")
