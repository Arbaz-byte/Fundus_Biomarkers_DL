"""Build and evaluate metadata-only baselines for shortcut analysis.

A shortcut test asks whether image metadata (for example resolution, color
statistics, or border size) can predict diabetic-retinopathy grades without
examining retinal pathology. 

Strong metadata-only performance suggests that an image model might exploit 
acquisition or framing cues instead of disease-related image content. 
These helpers prepare metadata features, construct an evaluation pipeline, 
and summarize its predictions.
"""
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Pathology labels
GRADE_LABELS = [0, 1, 2, 3, 4]
GRADE_NAMES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}

# Pure acquisition geometry: known before looking at any retinal content.
ACQUISITION_FEATURES = ["width", "height", "aspect_ratio"]

# Whole-image statistics  (include the black border).

WHOLE_IMAGE_FEATURES = [
    "file_size_bytes", "brightness_mean", "red_mean", "green_mean", "blue_mean",
    "red_std", "green_std", "blue_std", "blur_laplacian_variance", "black_border_fraction",
]

ALL_WHOLE_IMAGE_FEATURES = ACQUISITION_FEATURES + WHOLE_IMAGE_FEATURES


# Statistics measured inside the detected retinal disc at a fixed size.
FOV_FRAMING_FEATURES = ["fov_area_fraction", "fov_bbox_width_fraction", "fov_bbox_height_fraction"]


FOV_COLOUR_SHARPNESS_FEATURES = [
    "fov_brightness_mean", "fov_red_mean", "fov_green_mean", "fov_blue_mean",
    "fov_red_std", "fov_green_std", "fov_blue_std", "fov_blur_laplacian_variance",
]
FOV_FEATURES = FOV_FRAMING_FEATURES + FOV_COLOUR_SHARPNESS_FEATURES

# Statistics measured inside the whole image at a fixed size.
LOG_FEATURES = ("file_size_bytes", "blur_laplacian_variance", "fov_blur_laplacian_variance")


class MetadataPreprocessor(BaseEstimator, TransformerMixin):
    
    """Log-transform and cap metadata features using training-fold statistics.
    
    Selected skewed columns are clipped to nonnegative values and transformed
    with ``log1p``. Then every feature is clipped to lower and upper quantile
    bounds learned during ``fit``. Learning these bounds only from the fit
    data allows this transformer to be used in cross-validation without
    leaking information from a scoring fold.
    
    Args:
        feature_names: Names and order of the input columns. ``X`` is
            interpreted as a table with these columns.
        log_features: Feature names to log-transform before quantile capping.
            Defaults to ``LOG_FEATURES``.
        lower_quantile: Lower quantile used to fit each feature's clipping
            bound. Defaults to 0.01.
        upper_quantile: Upper quantile used to fit each feature's clipping
            bound. Defaults to 0.99.
            
    Attributes:
        lower_bounds_: Per-feature lower clipping bounds learned by ``fit``.
        upper_bounds_: Per-feature upper clipping bounds learned by ``fit``.
        n_features_in_: Number of input features observed during ``fit``.
    """
    
    def __init__(self, feature_names, log_features=LOG_FEATURES, lower_quantile=0.01, upper_quantile=0.99):
        
        self.feature_names = feature_names
        self.log_features = log_features
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        
    def _transform_frame(self, X):
        
        # convert X into a DataFrame and cast all columns as floats. copy() is necessary because we will modify
        frame = pd.DataFrame(X, columns=list(self.feature_names)).astype(float).copy()
        
        for column in self.log_features:
            if column in frame:
                frame[column] = np.log1p(frame[column].clip(lower=0))
                
        return frame
    
    def fit(self, X, y=None):
        """Learn per-feature clipping bounds from the supplied training data.
        
        Args:
            X: Two-dimensional feature data with columns in
                ``feature_names`` order.
            y: Ignored; present to follow the scikit-learn transformer API.
            
        Returns:
            This fitted transformer, with ``lower_bounds_``,
            ``upper_bounds_``, and ``n_features_in_`` set.
        """
        frame = self._transform_frame(X)
        
        # fit the quantiles to each column
        self.lower_bounds_ = frame.quantile(self.lower_quantile)
        self.upper_bounds_ = frame.quantile(self.upper_quantile)
        
        self.n_features_in_ = frame.shape[1]
        
        return self
    
    def transform(self, X):
        """Apply log transforms and the clipping bounds learned by ``fit``.
        
        Args:
            X: Two-dimensional feature data with columns in
                ``feature_names`` order.
                
        Returns:
            NumPy array of transformed features, with the same row and column
            counts as ``X``.
        """
        frame = self._transform_frame(X)
        return frame.clip(self.lower_bounds_, self.upper_bounds_, axis="columns").to_numpy()


def make_shortcut_pipeline(classifier, feature_names: Sequence[str]) -> Pipeline:
    
    """Create a scikit-learn pipeline for metadata-only classification.

    Processing order is log transformation and quantile capping, median
    imputation, standardization, and then classification. Keeping
    preprocessing inside the pipeline ensures its learned statistics are
    refit within each training fold rather than using held-out data.

    Args:
        classifier: Scikit-learn-compatible classifier implementing ``fit``
            and ``predict``.
        feature_names: Ordered names of the metadata columns supplied to the
            pipeline.

    Returns:
        A scikit-learn ``Pipeline`` with steps named ``"cap_and_log"``,
        ``"impute"``, ``"scale"``, and ``"classifier"``.
    """
    return Pipeline([
        ("cap_and_log", MetadataPreprocessor(feature_names=list(feature_names))),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("classifier", classifier),
    ])


def shortcut_metrics(y_true, y_pred) -> dict:
    """Compute classification metrics for the five diabetic-retinopathy grades.

    Quadratic weighted kappa is the primary ordinal metric; macro F1 and
    balanced accuracy help reveal performance across imbalanced classes,
    while accuracy reports the overall fraction of correct predictions.

    Args:
        y_true: One-dimensional sequence of true grade labels.
        y_pred: One-dimensional sequence of predicted grade labels aligned
            with ``y_true``.

    Returns:
        Dictionary with ``"qwk"``, ``"macro_f1"``,
        ``"balanced_accuracy"``, and ``"accuracy"`` metric values.
    """
    return {
        "qwk": cohen_kappa_score(y_true, y_pred, labels=GRADE_LABELS, weights="quadratic"),
        "macro_f1": f1_score(y_true, y_pred, labels=GRADE_LABELS, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def cross_validate_pipeline(pipeline, X: pd.DataFrame, y, cv_splits) -> pd.DataFrame:
    """Evaluate a pipeline on precomputed train/score folds.

    A fresh clone is fitted on each fold's training indices and evaluated on
    that fold's held-out indices. This reports fold-wise performance while
    allowing pipeline preprocessing to learn only from each fold's training
    rows.

    Args:
        pipeline: Scikit-learn-compatible estimator or pipeline supporting
            ``fit`` and ``predict``.
        X: Feature DataFrame. Fold indices select rows by position using
            ``iloc``.
        y: One-dimensional sequence of labels aligned by row position with
            ``X``.
        cv_splits: Iterable of ``(fit_indices, score_indices)`` pairs, such
            as the output of a scikit-learn cross-validation splitter.

    Returns:
        DataFrame with one row per fold, containing ``"fold"`` and all
        metrics returned by ``shortcut_metrics``.
    """
    rows = []
    y = np.asarray(y)

    for fold_number, (fit_indices, score_indices) in enumerate(cv_splits, start=1):

        model = clone(pipeline)
        model.fit(X.iloc[fit_indices], y[fit_indices])
        rows.append({"fold": fold_number, **shortcut_metrics(y[score_indices], model.predict(X.iloc[score_indices]))})
    return pd.DataFrame(rows)


def resolution_label(frame: pd.DataFrame) -> pd.Series:
    """Encode each image's dimensions as a resolution string.

    Args:
        frame: DataFrame containing numeric ``"width"`` and ``"height"``
            columns.

    Returns:
        String Series with the same index as ``frame``, formatted as
        ``"<width>x<height>"`` (for example, ``"2588x1958"``).
    """
    return frame["width"].astype(int).astype(str) + "x" + frame["height"].astype(int).astype(str)


def resolution_lookup_predictions(train: pd.DataFrame, target: pd.DataFrame, label_column: str = "diagnosis"):
    """Predict target labels using only each image's exact resolution.

    For each resolution present in the training data, the predicted label is
    its most frequent training label. Unseen resolutions use the overall
    training-label mode. This simple baseline tests whether acquisition
    resolution alone is associated with the target, without fitting a model
    or using target labels.

    Args:
        train: Training DataFrame containing ``"width"``, ``"height"``, and
            ``label_column``.
        target: DataFrame containing ``"width"`` and ``"height"`` for rows
            to predict.
        label_column: Name of the target-label column in ``train``.
            Defaults to ``"diagnosis"``.

    Returns:
        NumPy integer array of predictions in the same row order as
        ``target``.
    """
    train_resolution = resolution_label(train)
    
    modal_grade = train.groupby(train_resolution)[label_column].agg(lambda grades: grades.value_counts().idxmax())
    fallback = int(train[label_column].value_counts().idxmax())
    
    return resolution_label(target).map(modal_grade).fillna(fallback).astype(int).to_numpy()
