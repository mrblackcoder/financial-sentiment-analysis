"""Multi-Layer Perceptron (MLP) for text classification"""
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import time


class MLPModel:
    """Multi-Layer Perceptron classifier"""

    def __init__(self, hidden_layers=(100, 50), max_iter=500, random_state=42):
        """
        Initialize MLP model

        Args:
            hidden_layers: Tuple of hidden layer sizes
            max_iter: Maximum iterations
            random_state: Random seed
        """
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
            verbose=False
        )
        self.training_time = None
        self.cv_scores = None

    def train(self, X_train, y_train, cv_folds=5):
        """
        Train MLP model

        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of cross-validation folds

        Returns:
            Trained model
        """
        print(f"Training MLP with hidden layers {self.model.hidden_layer_sizes}...")

        # Cross-validation
        print(f"Performing {cv_folds}-fold cross-validation...")
        self.cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=cv_folds, scoring='f1_macro', n_jobs=-1
        )
        print(f"CV F1-Score: {self.cv_scores.mean():.4f} ± {self.cv_scores.std():.4f}")

        # Train on full training set
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time

        print(f"Training completed in {self.training_time:.2f}s")
        print(f"Training iterations: {self.model.n_iter_}")
        print(f"Training loss: {self.model.loss_:.4f}")

        return self.model

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)

    def get_model_info(self):
        """Get model information"""
        return {
            'name': 'Multi-Layer Perceptron (MLP)',
            'hidden_layers': self.model.hidden_layer_sizes,
            'num_parameters': sum([w.size for w in self.model.coefs_]) + sum([b.size for b in self.model.intercepts_]),
            'training_iterations': self.model.n_iter_,
            'training_loss': self.model.loss_,
            'training_time': self.training_time,
            'cv_scores': self.cv_scores
        }


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification

    # Generate dummy data
    X, y = make_classification(n_samples=500, n_features=100, n_classes=3,
                              n_informative=50, random_state=42)

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    mlp = MLPModel(hidden_layers=(100, 50))
    mlp.train(X_train, y_train)

    # Evaluate
    from sklearn.metrics import classification_report
    y_pred = mlp.predict(X_test)
    print("\nTest Results:")
    print(classification_report(y_test, y_pred))

    # Model info
    print("\nModel Info:")
    for key, value in mlp.get_model_info().items():
        print(f"  {key}: {value}")
