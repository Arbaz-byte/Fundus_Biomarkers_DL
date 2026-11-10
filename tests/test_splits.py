"""Tests for patient-aware train/validation/test splitting.

The project must avoid assigning images from the same patient to different
partitions, so this test creates repeated patient IDs and verifies that the
split operation and leakage guard maintain this constraint across all outputs.
"""

import pandas as pd

from fundus.splits import patient_split, assert_no_patient_leakage


def test_no_leakage():
    """Each patient should stay within one split and all three splits should be present."""
    df = pd.DataFrame({
        "path": [f"img{i}.png" for i in range(100)],
        "label": [0] * 100,
        "patient_id": [f"p{i // 2}" for i in range(100)],  # two images per patient
        "site": ["x"] * 100,
    })
    out = patient_split(df, 0.2, 0.2, seed=1)
    assert_no_patient_leakage(out)
    assert set(out["split"]) == {"train", "val", "test"}
