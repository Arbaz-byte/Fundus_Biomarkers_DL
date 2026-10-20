"""Evaluation metrics. Aggregate outputs only."""
import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score


def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def binary_auroc(y_true, y_score):
    return roc_auc_score(y_true, y_score)


def expected_calibration_error(y_true, probs, n_bins=10):
    """ECE for binary probabilities."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs > lo) & (probs <= hi)
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - probs[mask].mean())
    return float(ece)
