"""Evaluation metrics"""
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, classification_report,
    confusion_matrix, roc_auc_score
)
import numpy as np


def calculate_all_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate comprehensive evaluation metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'cohens_kappa': cohen_kappa_score(y_true, y_pred)
    }

    # Add ROC-AUC if probabilities provided
    if y_proba is not None:
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_proba,
                                                   multi_class='ovr', average='macro')
            metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_proba,
                                                   multi_class='ovo', average='macro')
        except:
            pass

    return metrics


def print_classification_metrics(metrics):
    """Pretty print classification metrics"""
    print("\n" + "="*60)
    print("CLASSIFICATION METRICS")
    print("="*60)

    print(f"\nOverall Performance:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  F1-Score (macro):   {metrics['f1_macro']:.4f}")
    print(f"  F1-Score (weighted):{metrics['f1_weighted']:.4f}")

    print(f"\nPrecision & Recall:")
    print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")

    print(f"\nStatistical Metrics:")
    print(f"  MCC:                {metrics['mcc']:.4f}")
    print(f"  Cohen's Kappa:      {metrics['cohens_kappa']:.4f}")

    if 'roc_auc_ovr' in metrics:
        print(f"\nROC-AUC:")
        print(f"  OvR (One-vs-Rest):  {metrics['roc_auc_ovr']:.4f}")
        print(f"  OvO (One-vs-One):   {metrics['roc_auc_ovo']:.4f}")

    print("="*60 + "\n")
