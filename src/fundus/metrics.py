"""
Scalar evaluation metrics for model predictions. These helpers compare labels with predicted classes or scores and return
aggregate metrics rather than per-sample results. They are useful for summarizing classification agreement, ranking performance, and probability
calibration on a held-out evaluation set.

"""
import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Measure agreement between ordered true and predicted labels.
    Disagreements are penalized quadratically, so predictions farther from
    the true class count more heavily than adjacent-class errors. This is
    useful for ordinal outcomes such as disease grades.
    
    Args:
        y_true: One-dimensional sequence of ground-truth class labels.
        y_pred: One-dimensional sequence of predicted class labels, aligned
            with ``y_true``.
            
    Returns:
        Quadratic weighted Cohen's kappa as a float. A value of 1 indicates
        perfect agreement; 0 indicates chance-level agreement.
        
    Raises:
        ValueError: If the input lengths or labels are invalid for
            ``sklearn.metrics.cohen_kappa_score``.
    """
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def binary_auroc(y_true, y_score):
    """
    
    Measure how well binary prediction scores rank positive examples.
    AUROC is the area under the receiver operating characteristic curve.
    It evaluates ranking across thresholds rather than accuracy at a single
    threshold, making it useful when comparing binary classifiers without
    first choosing a decision threshold.
    
    Args:
        y_true: One-dimensional sequence of binary ground-truth labels.
        y_score: One-dimensional sequence of continuous scores for the
            positive class, aligned with ``y_true``. These may be probabilities
            or other classifier scores.
            
    Returns:
        Area under the ROC curve as a float. 1.0 indicates perfect ranking;
        0.5 corresponds to chance-level ranking.
        
    Raises:
        ValueError: If the inputs are incompatible or ``y_true`` does not
            contain both binary classes.
    """
    return roc_auc_score(y_true, y_score)


def expected_calibration_error(y_true, probs, n_bins=10):
    
    """Summarize the calibration error of binary predicted probabilities.
    Probabilities are grouped into ``n_bins`` equal-width bins on ``[0, 1]``.
    
    For each non-empty bin, the absolute difference between the average
    observed positive rate and the average predicted probability is weighted
    by the fraction of all samples in that bin. Lower values indicate better
    calibration. 
    
    Calibration matters when predicted probabilities are used
    as confidence estimates or to support downstream decisions.
    The implementation uses left-open, right-closed bins, so a probability
    exactly equal to 0 is not included in any bin.

    Args:
        y_true: One-dimensional sequence of binary ground-truth labels
            (typically 0 or 1).
        probs: One-dimensional sequence of predicted positive-class
            probabilities in ``[0, 1]``, aligned with ``y_true``.
        n_bins: Number of equal-width probability bins. Defaults to 10.
        
    Returns:
        Expected calibration error as a Python float. A value of 0 indicates
        that predicted confidence matches the observed positive rate in every
        occupied bin.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs > lo) & (probs <= hi)
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - probs[mask].mean())
    return float(ece)
