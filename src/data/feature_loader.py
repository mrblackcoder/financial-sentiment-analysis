"""Feature loading utilities"""
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.helpers import load_pickle


def load_features(feature_type='tfidf', split='train'):
    """
    Load features and labels for a given split

    Args:
        feature_type: Type of features ('tfidf', 'word2vec', 'financial')
        split: Data split ('train', 'test', 'val')

    Returns:
        X, y: Features and labels
    """
    feature_dir = Path('data/features')

    # Try to load features
    feature_file = feature_dir / f'{split}_{feature_type}_features.pkl'
    label_file = feature_dir / f'{split}_labels.pkl'

    if not feature_file.exists() or not label_file.exists():
        # Generate dummy data for demo purposes
        print(f"Warning: Feature files not found. Generating dummy data for demo...")
        return generate_dummy_data(split)

    X = load_pickle(feature_file)
    y = load_pickle(label_file)

    return X, y


def generate_dummy_data(split='test'):
    """Generate dummy data for demonstration"""
    np.random.seed(42)

    # Generate dummy features and labels
    if split == 'test':
        n_samples = 75
    elif split == 'train':
        n_samples = 300
    else:
        n_samples = 100

    n_features = 609  # TF-IDF feature count

    # Random sparse features
    X = np.random.rand(n_samples, n_features) * 0.3
    # Random labels (0: Negative, 1: Neutral, 2: Positive)
    y = np.random.randint(0, 3, size=n_samples)

    # Make it roughly balanced
    y[:n_samples//3] = 0
    y[n_samples//3:2*n_samples//3] = 1
    y[2*n_samples//3:] = 2
    np.random.shuffle(y)

    return X, y
