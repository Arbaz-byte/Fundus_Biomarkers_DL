"""Tests for ``scripts/aptos_split_dataset.py``.

Run from the project root:
    pytest tests/test_aptos_split_dataset.py -v

The tests use temporary synthetic datasets for deterministic split, validation,
copy, symlink, overwrite, and leakage checks. A real-data consistency check is
skipped automatically when the APTOS files are unavailable.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import scripts.aptos_split_dataset as b  # noqa: E402

ROWS = [
    # id_code, diagnosis, split
    ("a1", 0, "train"),
    ("a2", 2, "train"),
    ("a3", 4, "train"),
    ("b1", 1, "val"),
    ("b2", 0, "val"),
    ("c1", 3, "test"),
    ("c2", 0, "test"),
]


def _make_project(tmp_path, rows=ROWS, extra_official=(), skip_images=()):
    """Create a temporary APTOS-like project structure.

    Args:
        tmp_path: pytest temporary directory for the synthetic project.
        rows: Manifest rows containing ``id_code``, ``diagnosis``, and ``split``.
        extra_official: Additional official training identifiers not present in
            the manifest.
        skip_images: Image identifiers whose synthetic files should be omitted.

    Returns:
        A tuple containing the manifest path, raw dataset directory, and output
        directory.
    """
    raw = tmp_path / "raw"
    (raw / "train_images").mkdir(parents=True)

    official = [(i, d) for i, d, _ in rows] + list(extra_official)
    pd.DataFrame(official, columns=["id_code", "diagnosis"]).to_csv(raw / "train.csv", index=False)

    for id_code, _ in official:
        if id_code not in skip_images:
            (raw / "train_images" / f"{id_code}.png").write_bytes(b"fake-" + id_code.encode())

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows, columns=["id_code", "diagnosis", "split"]).to_csv(manifest, index=False)
    return manifest, raw, tmp_path / "out"


# ----------------------------- happy path --------------------------------- #
def test_builds_folders_and_csvs(tmp_path):
    """Verify that building produces image folders and split CSV files."""
    manifest, raw, out = _make_project(tmp_path)
    b.build_split_dataset(manifest, raw, out)

    for split in b.SPLITS:
        assert (out / f"{split}_images").is_dir()
        assert (out / f"{split}.csv").is_file()


def test_csv_contents_match_manifest(tmp_path):
    """Verify each split CSV contains exactly its manifest identifiers and labels."""
    manifest, raw, out = _make_project(tmp_path)
    b.build_split_dataset(manifest, raw, out)

    for split in b.SPLITS:
        df = pd.read_csv(out / f"{split}.csv", dtype={"id_code": str})
        assert list(df.columns) == ["id_code", "diagnosis"]
        expected = {(i, d) for i, d, s in ROWS if s == split}
        assert set(zip(df["id_code"], df["diagnosis"])) == expected


def test_images_land_in_correct_split_folder(tmp_path):
    """Verify every image is copied only into its manifest-assigned folder."""
    manifest, raw, out = _make_project(tmp_path)
    b.build_split_dataset(manifest, raw, out)

    for id_code, _, split in ROWS:
        assert (out / f"{split}_images" / f"{id_code}.png").is_file()
        for other in set(b.SPLITS) - {split}:
            assert not (out / f"{other}_images" / f"{id_code}.png").exists()


def test_no_leakage_between_splits(tmp_path):
    """Verify returned split DataFrames contain disjoint identifier sets."""
    manifest, raw, out = _make_project(tmp_path)
    result = b.build_split_dataset(manifest, raw, out)

    ids = {s: set(df["id_code"]) for s, df in result.items()}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]


def test_image_bytes_are_copied_unchanged(tmp_path):
    """Verify copy mode preserves the source image bytes."""
    manifest, raw, out = _make_project(tmp_path)
    b.build_split_dataset(manifest, raw, out)
    assert (out / "train_images" / "a1.png").read_bytes() == b"fake-a1"


def test_ignores_official_ids_not_in_manifest(tmp_path):
    """Verify official identifiers outside the manifest are not materialized."""
    manifest, raw, out = _make_project(tmp_path, extra_official=[("zz", 1)])
    b.build_split_dataset(manifest, raw, out)
    for split in b.SPLITS:
        assert not (out / f"{split}_images" / "zz.png").exists()


def test_rerun_is_idempotent(tmp_path):
    """Verify rebuilding an existing output without overwrite remains valid."""
    manifest, raw, out = _make_project(tmp_path)
    b.build_split_dataset(manifest, raw, out)
    b.build_split_dataset(manifest, raw, out)  # must not raise
    assert len(list((out / "train_images").iterdir())) == 3


def test_symlink_mode(tmp_path):
    """Verify symlink mode creates symbolic links instead of copies."""
    manifest, raw, out = _make_project(tmp_path)
    b.build_split_dataset(manifest, raw, out, mode="symlink")
    assert (out / "train_images" / "a1.png").is_symlink()


def test_overwrite_removes_stale_images(tmp_path):
    """Verify stale files are detected and removed by overwrite mode."""
    manifest, raw, out = _make_project(tmp_path)
    b.build_split_dataset(manifest, raw, out)

    stale = out / "train_images" / "stale.png"
    stale.write_bytes(b"x")
    with pytest.raises(AssertionError):  # extra file is detected
        b.build_split_dataset(manifest, raw, out)

    b.build_split_dataset(manifest, raw, out, overwrite=True)
    assert not stale.exists()


# ----------------------------- failure cases ------------------------------ #
def test_missing_manifest_raises(tmp_path):
    """Verify a missing manifest raises FileNotFoundError."""
    _, raw, out = _make_project(tmp_path)
    with pytest.raises(FileNotFoundError):
        b.build_split_dataset(tmp_path / "nope.csv", raw, out)


def test_missing_image_fails_before_writing(tmp_path):
    """Verify missing source images stop the build before output is created."""
    manifest, raw, out = _make_project(tmp_path, skip_images={"b1"})
    with pytest.raises(FileNotFoundError, match="missing"):
        b.build_split_dataset(manifest, raw, out)
    assert not out.exists()  # nothing was written


def test_id_not_in_official_train_raises(tmp_path):
    """Verify manifest identifiers must exist in the official training CSV."""
    rows = ROWS + [("ghost", 0, "test")]
    manifest, raw, out = _make_project(tmp_path)
    pd.DataFrame(rows, columns=["id_code", "diagnosis", "split"]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="not in the official train.csv"):
        b.build_split_dataset(manifest, raw, out)


def test_label_mismatch_raises(tmp_path):
    """Verify manifest diagnoses must match official training labels."""
    manifest, raw, out = _make_project(tmp_path)
    bad = [(i, (d + 1) % 5 if i == "a1" else d, s) for i, d, s in ROWS]
    pd.DataFrame(bad, columns=["id_code", "diagnosis", "split"]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="disagree"):
        b.build_split_dataset(manifest, raw, out)


def test_duplicate_id_in_manifest_raises(tmp_path):
    """Verify duplicate identifiers are rejected to prevent data leakage."""
    manifest, raw, out = _make_project(tmp_path)
    dup = ROWS + [("a1", 0, "val")]
    pd.DataFrame(dup, columns=["id_code", "diagnosis", "split"]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        b.build_split_dataset(manifest, raw, out)


def test_unknown_split_value_raises(tmp_path):
    """Verify manifest split names outside train, val, and test are rejected."""
    manifest, raw, out = _make_project(tmp_path)
    bad = [(i, d, "holdout" if i == "c1" else s) for i, d, s in ROWS]
    pd.DataFrame(bad, columns=["id_code", "diagnosis", "split"]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="unknown split"):
        b.build_split_dataset(manifest, raw, out)


def test_invalid_diagnosis_raises(tmp_path):
    """Verify diagnoses outside the valid 0--4 range are rejected."""
    manifest, raw, out = _make_project(tmp_path)
    bad = [(i, 7 if i == "a1" else d, s) for i, d, s in ROWS]
    pd.DataFrame(bad, columns=["id_code", "diagnosis", "split"]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="invalid diagnosis"):
        b.build_split_dataset(manifest, raw, out)


def test_missing_column_raises(tmp_path):
    """Verify manifests missing required columns raise a validation error."""
    manifest, raw, out = _make_project(tmp_path)
    pd.read_csv(manifest).drop(columns=["split"]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        b.build_split_dataset(manifest, raw, out)


# ----------------------- real data sanity check ------------------ #
REAL_MANIFEST = Path("configs/splits/aptos_official_split_v1.csv")
REAL_RAW = Path("data/raw/APTOS-2019-kaggle-competition")


@pytest.mark.skipif(
    not (REAL_MANIFEST.is_file() and (REAL_RAW / "train.csv").is_file()),
    reason="Real APTOS data not found (run pytest from the project root).",
)
def test_real_manifest_is_consistent_with_official_train():
    """Verify real project data agrees with the official manifest and images."""
    manifest = b.load_manifest(REAL_MANIFEST)
    official = b.load_official_train(REAL_RAW / "train.csv")
    b.check_manifest_against_official(manifest, official)

    assert set(manifest["split"]) == set(b.SPLITS)
    assert len(manifest) == len(official)  # manifest covers the whole official train set

    missing = [i for i in manifest["id_code"] if b.locate_image(REAL_RAW / "train_images", i) is None]
    assert not missing, f"{len(missing)} manifest images missing on disk, e.g. {missing[:5]}"
