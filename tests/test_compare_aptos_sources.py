"""Tests for comparing legacy APTOS data with the official training source.

The test builds small temporary CSV and image-file fixtures(test setup data or resources prepared so a test can run in a controlled, repeatable way.) to check ID
inventory, diagnosis agreement, exact-byte hashes, per-ID report output, and
the guarantee that official test-image files are not accessed.
"""

import csv

from scripts.compare_aptos_sources import compare_sources


def _write_csv(path, fieldnames, rows):
    """Create a CSV fixture with the requested columns and records.
    Args:
        path: Destination path for the temporary CSV.
        fieldnames: Column names to write as the header.
        rows: Dictionaries containing the CSV records.

    This helper keeps test setup concise while producing the same CSV structure
    expected by the source-comparison code.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_image_bytes(path, content):
    """Write known bytes to a temporary image path for hash comparisons.
    
    Args:
        path: Destination path, including the image filename.
        content: Byte sequence to store as the file contents.
        
    Controlled file contents let the test verify identical and differing
    SHA-256 results without requiring real image assets.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_compare_sources_checks_ids_labels_hashes_and_unlabeled_test(tmp_path):
    """Verify source comparison results and ensure official test files stay unopened.
    
    The fixture contains shared, source-only, label-mismatched, byte-identical,
    and byte-different training examples. The assertions cover the aggregate
    summary and generated per-ID CSV, while confirming that the official test
    CSV is treated as unlabeled and no official test image is opened or hashed.
    """
    old_root = tmp_path / "old"
    new_root = tmp_path / "official"
    
    old_rows = {
        "train_1.csv": [
            {"id_code": "same", "diagnosis": "2"},
            {"id_code": "label-diff", "diagnosis": "1"},
            {"id_code": "old-only", "diagnosis": "4"},
        ],
        "valid.csv": [],
        "test.csv": [],
    }
    for csv_name, rows in old_rows.items():
        _write_csv(
            old_root / csv_name,
            ["id_code", "diagnosis"],
            rows,
        )

    old_image_dirs = {
        "train_1.csv": "train_images/train_images",
        "valid.csv": "val_images/val_images",
        "test.csv": "test_images/test_images",
    }
    _write_image_bytes(old_root / old_image_dirs["train_1.csv"] / "same.png", b"same")
    _write_image_bytes(
        old_root / old_image_dirs["train_1.csv"] / "label-diff.png",
        b"old bytes",
    )
    _write_image_bytes(
        old_root / old_image_dirs["train_1.csv"] / "old-only.png",
        b"old only",
    )
    for image_dir in old_image_dirs.values():
        (old_root / image_dir).mkdir(parents=True, exist_ok=True)

    _write_csv(
        new_root / "train.csv",
        ["id_code", "diagnosis"],
        [
            {"id_code": "same", "diagnosis": "2"},
            {"id_code": "label-diff", "diagnosis": "3"},
            {"id_code": "new-only", "diagnosis": "0"},
        ],
    )
    _write_image_bytes(new_root / "train_images/same.png", b"same")
    _write_image_bytes(new_root / "train_images/label-diff.png", b"new bytes")
    _write_image_bytes(new_root / "train_images/new-only.png", b"new only")
    _write_csv(new_root / "test.csv", ["id_code"], [{"id_code": "sealed-test"}])
    report_path = tmp_path / "processed/comparison.csv"
    summary = compare_sources(old_root, new_root, report_path)
    
    assert summary["id_comparison"]["common_ids"] == 2
    assert summary["id_comparison"]["old_only_ids"] == 1
    assert summary["id_comparison"]["new_train_only_ids"] == 1
    assert summary["label_comparison"]["matching_labels"] == 1
    assert summary["label_comparison"]["mismatching_labels"] == 1
    assert summary["file_hash_comparison"]["identical_hashes"] == 1
    assert summary["file_hash_comparison"]["different_hashes"] == 1
    assert summary["official_test_csv"]["has_diagnosis_column"] is False
    assert summary["official_test_csv"]["test_images_opened_or_hashed"] is False
    assert report_path.is_file()
    
    with report_path.open(newline="", encoding="utf-8") as file:
        report = {row["id_code"]: row for row in csv.DictReader(file)}
    assert report["same"]["byte_hash_match"] == "True"
    assert report["label-diff"]["label_match"] == "False"
