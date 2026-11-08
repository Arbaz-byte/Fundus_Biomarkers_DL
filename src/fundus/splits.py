"""Assign dataset rows to train, validation, and test splits by patient.

The split helpers operate on pandas dataframes containing one row per image.
All images with the same patient ID receive the same split, preventing
patient-specific information from leaking between model training and
evaluation.
"""
import numpy as np
import pandas as pd


def patient_split(df: pd.DataFrame, val_fraction=0.15, test_fraction=0.15, seed=42):
    
    """Return a copy of an image manifest with a patient-level split column.

    Unique patient IDs are shuffled using ``seed``. The first ``int(number_of_patients * test_fraction)`` patients are assigned to
    ``"test"``, the next ``int(number_of_patients * val_fraction)`` to ``"val"``, and all remaining patients to ``"train"``. Fractions therefore
    apply to patients (not image rows), and fractional counts are rounded down. The input row order and index are preserved.

    Args:
        df: Image manifest with a ``"patient_id"`` column.
        val_fraction: Fraction of unique patients to reserve for validation.
            Defaults to 0.15.
        test_fraction: Fraction of unique patients to reserve for testing.
            Defaults to 0.15.
        seed: Seed for the random generator, making assignments reproducible.
            Defaults to 42.

    Returns:
        A copy of ``df`` with a ``"split"`` column containing ``"train"``,
        ``"val"``, or ``"test"``. If ``df`` already has a ``"split"`` column,
        its values are replaced.

    Raises:
        ValueError: If ``df`` does not contain a ``"patient_id"`` column.
    """
    if "patient_id" not in df.columns:
        raise ValueError("patient_id column required")

    # Check for unique patient Ids
    rng = np.random.default_rng(seed)
    patients = df["patient_id"].unique().to_numpy(copy=True)

    # Shuffle
    rng.shuffle(patients)
    n = len(patients)

    # Number of train and validation images
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)

    # select unique number of patient IDs for test and validation
    test_p = set(patients[:n_test])
    val_p = set(patients[n_test:n_test + n_val])

    # Map patient ID with test and validation
    split = df["patient_id"].map(
        lambda p: "test" if p in test_p else ("val" if p in val_p else "train")
    )

    # Final data frame
    out = df.copy()
    out["split"] = split
    return out


def assert_no_patient_leakage(df: pd.DataFrame):
    
    """Raise if any patient appears in more than one dataset split.

    This is a validation check for a split manifest: all rows for each patient
    must have at most one distinct value in the ``"split"`` column. It helps catch
    leakage that could make model evaluation overly optimistic.

    Args:
        df: Image manifest containing ``"patient_id"`` and ``"split"`` columns.

    Raises:
        AssertionError: If one or more patients occur in multiple split values.
        KeyError: If either required column is missing.

    Returns:
        None.
    """
    groups = df.groupby("patient_id")["split"].nunique()
    bad = groups[groups > 1]
    if len(bad):
        raise AssertionError(f"{len(bad)} patients appear in more than one split")
