"""Tests for shortcut baselines and duplicate detection.

These checks cover the metadata-based shortcut model, a small cross-validation
helper for evaluation, and the duplicate-group utility used to prevent
near-duplicate images from leaking across dataset splits.
"""

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

from fundus.duplicates import build_duplicate_groups
from fundus.features.shortcut import (
    MetadataPreprocessor,
    cross_validate_pipeline,
    make_shortcut_pipeline,
    resolution_lookup_predictions,
    shortcut_metrics,
)


def test_preprocessor_learns_bounds_from_fit_data_only():
    """The metadata scaler should clip any future feature to training-derived bounds."""
    train = pd.DataFrame({"a": np.arange(100.0), "file_size_bytes": np.arange(100.0) * 1000})
    pre = MetadataPreprocessor(feature_names=["a", "file_size_bytes"]).fit(train)
    new = pd.DataFrame({"a": [1e6], "file_size_bytes": [1e12]})
    out = pre.transform(new)
    assert out[0, 0] == pre.upper_bounds_["a"]          # capped at the training 99th percentile
    assert out[0, 1] == pre.upper_bounds_["file_size_bytes"]


def test_pipeline_runs_and_cv_has_one_row_per_fold():
    """The cross-validation wrapper should emit one evaluation row per fold."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 2)), columns=["x", "y"])
    y = np.tile([0, 1, 2], 20)
    cv = list(StratifiedKFold(3, shuffle=True, random_state=0).split(X, y))
    table = cross_validate_pipeline(make_shortcut_pipeline(DummyClassifier(strategy="most_frequent"), ["x", "y"]), X, y, cv)
    assert len(table) == 3 and {"qwk", "macro_f1", "balanced_accuracy", "accuracy"} <= set(table.columns)


def test_shortcut_metrics_perfect_prediction():
    """Perfect predictions should produce the maximum quadratic weighted kappa."""
    y = np.array([0, 1, 2, 3, 4])
    assert shortcut_metrics(y, y)["qwk"] == 1.0


def test_resolution_lookup_uses_training_labels_only():
    """Unseen resolutions should fall back to the overall training mode instead of guessing."""
    train = pd.DataFrame({"width": [10, 10, 20, 20], "height": [5, 5, 5, 5], "diagnosis": [0, 0, 3, 3]})
    target = pd.DataFrame({"width": [10, 20, 99], "height": [5, 5, 5]})
    assert list(resolution_lookup_predictions(train, target)) == [0, 3, 0]  # unseen resolution -> overall mode


def test_duplicate_groups_link_copies_only(tmp_path):
    """Only visually near-identical files should be grouped as duplicates."""
    rng = np.random.default_rng(1)
    base = (rng.random((200, 200, 3)) * 255).astype(np.uint8)
    Image.fromarray(base).save(tmp_path / "a.png")
    Image.fromarray(np.clip(base.astype(int) + 2, 0, 255).astype(np.uint8)).save(tmp_path / "a_copy.png")
    Image.fromarray((rng.random((200, 200, 3)) * 255).astype(np.uint8)).save(tmp_path / "other.png")
    groups, links = build_duplicate_groups([tmp_path / "a.png", tmp_path / "a_copy.png", tmp_path / "other.png"])
    assert groups[0] == groups[1] and groups[2] != groups[0]
    assert len(links) == 1
