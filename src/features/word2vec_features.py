"""Word2Vec Feature Extraction

Word embedding based feature extraction for financial text.
Uses pre-trained or custom trained Word2Vec models.

Note: For this project, TF-IDF was chosen as the primary method
because financial sentiment relies heavily on keyword presence
(e.g., 'profit', 'loss', 'surge') rather than semantic context.
This module is provided for completeness and future experiments.
"""

import numpy as np
from typing import List, Optional
import warnings

try:
    from gensim.models import Word2Vec
    from gensim.models import KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    warnings.warn("gensim not installed. Word2Vec features unavailable.")


class Word2VecExtractor:
    """
    Word2Vec feature extractor for financial text.

    Creates dense word embedding vectors by averaging word vectors.
    Can use pre-trained models (Google News, GloVe) or train custom.

    Parameters:
    -----------
    vector_size : int
        Dimensionality of word vectors (default: 100)
    window : int
        Context window size (default: 5)
    min_count : int
        Minimum word frequency (default: 2)
    pretrained_path : str, optional
        Path to pre-trained model (e.g., GoogleNews vectors)
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        pretrained_path: Optional[str] = None
    ):
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim is required for Word2Vec features")

        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.pretrained_path = pretrained_path
        self.model = None
        self.is_fitted = False

    def fit(self, texts: List[str]) -> 'Word2VecExtractor':
        """
        Train Word2Vec model on texts or load pre-trained.

        Parameters:
        -----------
        texts : List[str]
            Training documents

        Returns:
        --------
        self
        """
        if self.pretrained_path:
            # Load pre-trained model
            try:
                self.model = KeyedVectors.load_word2vec_format(
                    self.pretrained_path, binary=True
                )
                self.vector_size = self.model.vector_size
            except Exception as e:
                warnings.warn(f"Could not load pretrained model: {e}")
                self._train_custom(texts)
        else:
            self._train_custom(texts)

        self.is_fitted = True
        return self

    def _train_custom(self, texts: List[str]):
        """Train custom Word2Vec on provided texts."""
        # Tokenize texts
        tokenized = [text.lower().split() for text in texts]

        # Train model
        self.model = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            seed=42
        )

    def _text_to_vector(self, text: str) -> np.ndarray:
        """Convert single text to averaged word vector."""
        words = text.lower().split()
        vectors = []

        for word in words:
            try:
                if hasattr(self.model, 'wv'):
                    vec = self.model.wv[word]
                else:
                    vec = self.model[word]
                vectors.append(vec)
            except KeyError:
                # Word not in vocabulary
                continue

        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.vector_size)

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to averaged word vectors.

        Parameters:
        -----------
        texts : List[str]
            Documents to transform

        Returns:
        --------
        np.ndarray
            Dense feature matrix (n_samples, vector_size)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return np.array([self._text_to_vector(text) for text in texts])

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)

    def get_n_features(self) -> int:
        """Get number of features (vector dimensionality)."""
        return self.vector_size


def extract_word2vec_features(
    train_texts: List[str],
    test_texts: List[str],
    val_texts: Optional[List[str]] = None,
    vector_size: int = 100
) -> dict:
    """
    Extract Word2Vec features for all splits.

    Parameters:
    -----------
    train_texts : List[str]
        Training documents
    test_texts : List[str]
        Test documents
    val_texts : List[str], optional
        Validation documents
    vector_size : int
        Word vector dimensionality

    Returns:
    --------
    dict with keys: 'train', 'test', 'val', 'extractor'
    """
    extractor = Word2VecExtractor(vector_size=vector_size)

    result = {
        'train': extractor.fit_transform(train_texts),
        'test': extractor.transform(test_texts),
        'extractor': extractor,
        'n_features': extractor.get_n_features()
    }

    if val_texts is not None:
        result['val'] = extractor.transform(val_texts)

    return result


# Comparison with TF-IDF for Financial Text
"""
WHY TF-IDF WAS CHOSEN OVER WORD2VEC FOR THIS PROJECT:

1. Financial Sentiment is Keyword-Based:
   - Words like 'profit', 'loss', 'surge', 'crash' directly indicate sentiment
   - TF-IDF weights rare/important words higher (exactly what we need)
   - Word2Vec captures semantic similarity but may miss keyword importance

2. Sparse vs Dense Representations:
   - TF-IDF: Sparse, interpretable, fast
   - Word2Vec: Dense, requires averaging which loses word-level importance

3. Training Data Size:
   - Word2Vec needs large corpora for good embeddings
   - Our dataset (3,761 samples) is relatively small
   - TF-IDF works well with smaller datasets

4. Performance:
   - TF-IDF achieved 96.81% F1-Score with MLP
   - This is already excellent for sentiment analysis
   - Word2Vec might not significantly improve results

FUTURE WORK:
- Try FinBERT (pre-trained on financial text)
- Combine TF-IDF + Word2Vec features
- Use domain-specific pre-trained embeddings
"""


if __name__ == "__main__":
    if GENSIM_AVAILABLE:
        # Test Word2Vec extractor
        sample_texts = [
            "Stock prices surged after strong earnings report",
            "Market crashed due to economic concerns",
            "Company maintains steady growth outlook"
        ]

        extractor = Word2VecExtractor(vector_size=50)
        features = extractor.fit_transform(sample_texts)

        print(f"Word2Vec Features shape: {features.shape}")
        print(f"Vector size: {extractor.get_n_features()}")
    else:
        print("gensim not available - Word2Vec features disabled")
