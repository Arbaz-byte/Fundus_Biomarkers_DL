"""

Helpers for the metadata-only shortcut test.

A shortcut test asks: can the DR grade be predicted from image metadata alone (resolution, colour
statistics, border size ...)? If yes, an image model could exploit the same cues instead of pathology.
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
    """
    Log-transform skewed columns, then cap every column at fit-time quantiles.

    Everything is learned in ``fit`` so it can be used inside a cross-validation pipeline without
    leaking information from the scoring fold.
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
        frame = self._transform_frame(X)
        
        # fit the quantiles to each column
        self.lower_bounds_ = frame.quantile(self.lower_quantile)
        self.upper_bounds_ = frame.quantile(self.upper_quantile)
        
        self.n_features_in_ = frame.shape[1]
        
        return self
    
    
    def transform(self, X):
        
        frame = self._transform_frame(X)
        return frame.clip(self.lower_bounds_, self.upper_bounds_, axis="columns").to_numpy()


def make_shortcut_pipeline(classifier, feature_names: Sequence[str]) -> Pipeline:
    
    """
    Log/cap -> median imputation -> standardisation -> classifier.
    
    """
    return Pipeline([
        ("cap_and_log", MetadataPreprocessor(feature_names=list(feature_names))),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("classifier", classifier),
    ])


def shortcut_metrics(y_true, y_pred) -> dict:
    
    """
    
    QWK (primary, grades are ordinal) plus metrics that stay honest under class imbalance.
    
    """
    return {
        "qwk": cohen_kappa_score(y_true, y_pred, labels=GRADE_LABELS, weights="quadratic"),
        "macro_f1": f1_score(y_true, y_pred, labels=GRADE_LABELS, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def cross_validate_pipeline(pipeline, X: pd.DataFrame, y, cv_splits) -> pd.DataFrame:
    
    """Fit a fresh clone on each fold's fitting rows and score it on the held-out rows."""
    rows = []
    y = np.asarray(y)
    
    for fold_number, (fit_indices, score_indices) in enumerate(cv_splits, start=1):
        
        model = clone(pipeline)
        model.fit(X.iloc[fit_indices], y[fit_indices])
        rows.append({"fold": fold_number, **shortcut_metrics(y[score_indices], model.predict(X.iloc[score_indices]))})
    return pd.DataFrame(rows)


def resolution_label(frame: pd.DataFrame) -> pd.Series:
    
    """Exact resolution as a string, e.g. '2588x1958'."""
    
    return frame["width"].astype(int).astype(str) + "x" + frame["height"].astype(int).astype(str)


def resolution_lookup_predictions(train: pd.DataFrame, target: pd.DataFrame, label_column: str = "diagnosis"):
    
    """ predict , for each image, the most common training grade at the same resolution.

    A plain lookup table with no learning. If this scores well, resolution alone leaks the grade.
    Resolutions never seen in training fall back to the overall most common training grade.
    """
    train_resolution = resolution_label(train)
    
    modal_grade = train.groupby(train_resolution)[label_column].agg(lambda grades: grades.value_counts().idxmax())
    fallback = int(train[label_column].value_counts().idxmax())
    
    return resolution_label(target).map(modal_grade).fillna(fallback).astype(int).to_numpy()
