"""LSTM/BiLSTM for text classification

This module provides LSTM implementation. To use it, install TensorFlow:
    pip install tensorflow

For systems without TensorFlow, use MLP model instead.
"""

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print(" ï¸  TensorFlow not available. Install with: pip install tensorflow")

import numpy as np
import time


class LSTMModel:
    """LSTM-based text classifier"""

    def __init__(self, vocab_size=10000, embedding_dim=100,
                 lstm_units=64, bidirectional=True,
                 dropout=0.3, num_classes=3, random_state=42):
        """
        Initialize LSTM model

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            lstm_units: Number of LSTM units
            bidirectional: Whether to use BiLSTM
            dropout: Dropout rate
            num_classes: Number of output classes
            random_state: Random seed
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_classes = num_classes
        self.random_state = random_state

        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        self.model = None
        self.history = None
        self.training_time = None

    def build_model(self, max_sequence_length=100):
        """Build LSTM model architecture"""

        model = models.Sequential([
            # Embedding layer
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=max_sequence_length
            ),

            # Dropout
            layers.Dropout(self.dropout),

            # LSTM layer(s)
            layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True))
            if self.bidirectional
            else layers.LSTM(self.lstm_units, return_sequences=True),

            layers.Bidirectional(layers.LSTM(self.lstm_units // 2))
            if self.bidirectional
            else layers.LSTM(self.lstm_units // 2),

            # Dropout
            layers.Dropout(self.dropout),

            # Dense layers
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout / 2),

            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None,
             epochs=50, batch_size=32, verbose=1):
        """
        Train LSTM model

        Args:
            X_train: Training sequences (integer encoded)
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Verbosity level

        Returns:
            Training history
        """
        if self.model is None:
            max_len = X_train.shape[1] if len(X_train.shape) > 1 else 100
            self.build_model(max_sequence_length=max_len)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=3,
                verbose=1
            )
        ]

        # Validation data
        validation_data = (X_val, y_val) if X_val is not None else None

        print(f"Training {'Bi' if self.bidirectional else ''}LSTM model...")
        print(f"Model parameters: {self.model.count_params():,}")

        # Train
        start_time = time.time()
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        self.training_time = time.time() - start_time

        print(f"Training completed in {self.training_time:.2f}s")

        return self.history

    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict(X, verbose=0)

    def get_model_info(self):
        """Get model information"""
        return {
            'name': f"{'Bidirectional ' if self.bidirectional else ''}LSTM",
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'num_parameters': self.model.count_params() if self.model else 0,
            'training_time': self.training_time,
            'epochs_trained': len(self.history.history['loss']) if self.history else 0
        }

    def plot_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return

        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history.history:
            ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        return fig


# Fallback for systems without TensorFlow
class LSTMModelFallback:
    """Fallback when TensorFlow is not available"""

    def __init__(self, *args, **kwargs):
        print(" ï¸  TensorFlow not installed. Using MLP as fallback.")
        from .mlp_model import MLPModel
        self.model = MLPModel()

    def train(self, X_train, y_train, **kwargs):
        return self.model.train(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# Use fallback if TensorFlow not available
if not TENSORFLOW_AVAILABLE:
    LSTMModel = LSTMModelFallback


# Example usage
if __name__ == "__main__":
    if TENSORFLOW_AVAILABLE:
        # Generate dummy sequential data
        vocab_size = 1000
        max_len = 50
        num_samples = 500

        X = np.random.randint(0, vocab_size, size=(num_samples, max_len))
        y = np.random.randint(0, 3, size=num_samples)

        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train
        lstm = LSTMModel(vocab_size=vocab_size, embedding_dim=64, lstm_units=32)
        lstm.train(X_train, y_train, epochs=10, batch_size=32)

        # Evaluate
        y_pred = lstm.predict(X_test)
        from sklearn.metrics import classification_report
        print("\nTest Results:")
        print(classification_report(y_test, y_pred))

        # Plot history
        lstm.plot_history()
    else:
        print("Install TensorFlow to use LSTM: pip install tensorflow")
