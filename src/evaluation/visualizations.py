"""Visualization functions"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(y_true, y_pred, class_names=None, title='Confusion Matrix',
                         figsize=(8, 6), cmap='Blues'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()

    return plt.gcf()


def plot_roc_curves(y_true, y_proba, class_names=None, title='ROC Curves',
                   figsize=(10, 8)):
    """Plot ROC curves for multi-class classification"""
    n_classes = y_proba.shape[1]

    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]

    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot
    plt.figure(figsize=figsize)
    colors = ['blue', 'green', 'red', 'orange', 'purple']

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5000)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def plot_feature_importance(importances, n_top=20, feature_names=None,
                           title='Feature Importance', figsize=(10, 8)):
    """Plot feature importance"""
    n_features = len(importances)

    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(n_features)]

    # Get top features
    indices = np.argsort(importances)[-n_top:][::-1]
    top_importances = importances[indices]
    top_names = [feature_names[i] for i in indices]

    # Plot
    plt.figure(figsize=figsize)
    plt.barh(range(n_top), top_importances, alpha=0.8)
    plt.yticks(range(n_top), top_names)
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    return plt.gcf()
