"""Patient-level splitting. Images of one patient never cross splits."""
import numpy as np
import pandas as pd


def patient_split(df: pd.DataFrame, val_fraction=0.15, test_fraction=0.15, seed=42):
    if "patient_id" not in df.columns:
        raise ValueError("patient_id column required")
    rng = np.random.default_rng(seed)
    patients = df["patient_id"].unique()
    rng.shuffle(patients)
    n = len(patients)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    test_p = set(patients[:n_test])
    val_p = set(patients[n_test:n_test + n_val])
    split = df["patient_id"].map(
        lambda p: "test" if p in test_p else ("val" if p in val_p else "train")
    )
    out = df.copy()
    out["split"] = split
    return out


def assert_no_patient_leakage(df: pd.DataFrame):
    groups = df.groupby("patient_id")["split"].nunique()
    bad = groups[groups > 1]
    if len(bad):
        raise AssertionError(f"{len(bad)} patients appear in more than one split")
