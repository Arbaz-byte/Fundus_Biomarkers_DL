import pandas as pd

from fundus.splits import patient_split, assert_no_patient_leakage


def test_no_leakage():
    df = pd.DataFrame({
        "path": [f"img{i}.png" for i in range(100)],
        "label": [0] * 100,
        "patient_id": [f"p{i // 2}" for i in range(100)],  # two images per patient
        "site": ["x"] * 100,
    })
    out = patient_split(df, 0.2, 0.2, seed=1)
    assert_no_patient_leakage(out)
    assert set(out["split"]) == {"train", "val", "test"}
