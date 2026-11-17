"""Tests for auditing the legacy and official APTOS dataset sources.

The test uses temporary CSVs and small generated images to verify metadata
profiling, source-comparison consistency, audit summaries and output files.
It also checks that the official test-image files are not opened or hashed.
"""

import csv
import hashlib

from PIL import Image

from scripts.audit import audit_outputs_exist, run_dataset_audit
from scripts.compare_aptos_sources import REPORT_FIELDS


def _write_csv(path, fieldnames, rows):
    """Write a CSV fixture with the given schema and records.
    
    Args:
        path: Destination path for the temporary CSV file.
        fieldnames: Column names to write in the header.
        rows: Dictionaries representing the CSV data rows.
        
    Creates parent directories as needed so tests can define a realistic
    dataset layout without depending on repository data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_image(path, size, color):
    """Create a small RGB image fixture and return its encoded file bytes.
    
    Args:
        path: Destination path for the image.
        size: Image dimensions as a ``(width, height)`` pair.
        color: RGB color used to fill the image.
        
    Returns:
        The saved image's exact bytes, used to construct matching SHA-256
        values for the source-comparison report.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path.read_bytes()


def test_audit_profiles_sources_and_keeps_official_test_images_unopened(tmp_path):
    """Check audit summaries, per-image outputs, and protected test-image handling.
    
    Builds minimal legacy and official datasets with two labeled images, plus
    an official test image containing invalid image bytes. A successful audit
    demonstrates that train-source metadata and distributions are generated,
    the supplied comparison report is accepted, all expected output files are
    written, and the official test image remains unopened and unhashed.
    """
    old_root = tmp_path / "old"
    official_root = tmp_path / "official"
    output_dir = tmp_path / "processed/audit"
    
    old_rows = {
        "train_1.csv": [{"id_code": "grade-zero", "diagnosis": "0"}],
        "valid.csv": [{"id_code": "grade-four", "diagnosis": "4"}],
        "test.csv": [],
    }
    old_image_dirs = {
        "train_1.csv": "train_images/train_images",
        "valid.csv": "val_images/val_images",
        "test.csv": "test_images/test_images",
    }
    for csv_name, rows in old_rows.items():
        _write_csv(old_root / csv_name, ["id_code", "diagnosis"], rows)
        (old_root / old_image_dirs[csv_name]).mkdir(parents=True, exist_ok=True)
        
    zero_bytes = _write_image(
        old_root / old_image_dirs["train_1.csv"] / "grade-zero.png",
        (8, 8),
        (30, 80, 120),
    )
    four_bytes = _write_image(
        old_root / old_image_dirs["valid.csv"] / "grade-four.png",
        (16, 8),
        (80, 120, 30),
    )
    
    _write_csv(
        official_root / "train.csv",
        ["id_code", "diagnosis"],
        [
            {"id_code": "grade-zero", "diagnosis": "0"},
            {"id_code": "grade-four", "diagnosis": "4"},
        ],
    )
    _write_image(official_root / "train_images/grade-zero.png", (8, 8), (30, 80, 120))
    _write_image(official_root / "train_images/grade-four.png", (16, 8), (80, 120, 30))
    _write_csv(
        official_root / "test.csv",
        ["id_code"],
        [{"id_code": "sealed-test"}],
    )
    (official_root / "test_images").mkdir(parents=True)
    (official_root / "test_images" / "must_not_be_opened.png").write_bytes(b"not an image")
    
    comparison_rows = []
    for image_id, diagnosis, image_bytes, split in [
        ("grade-zero", 0, zero_bytes, "train"),
        ("grade-four", 4, four_bytes, "val"),
    ]:
        digest = hashlib.sha256(image_bytes).hexdigest()
        comparison_rows.append(
            {
                "id_code": image_id,
                "old_split": split,
                "old_diagnosis": diagnosis,
                "new_diagnosis": diagnosis,
                "label_match": True,
                "old_sha256": digest,
                "new_sha256": digest,
                "byte_hash_match": True,
            }
        )
    report_path = tmp_path / "processed/comparison.csv"
    _write_csv(report_path, REPORT_FIELDS, comparison_rows)

    result = run_dataset_audit(
        old_root,
        official_root,
        report_path,
        output_dir,
        include_fov=False,
    )

    summary = result["summary"]
    assert summary["audit_complete"] is True
    assert summary["image_inventory"]["old_copy_profiled_images"] == 2
    assert summary["image_inventory"]["official_train_profiled_images"] == 2
    assert summary["source_comparison_report"]["identical_byte_hashes"] == 2
    assert summary["official_test_handling"]["csv_columns"] == ["id_code"]
    assert summary["official_test_handling"]["test_images_opened_or_hashed"] is False
    assert summary["csv_schemas"]["official_train"]["train.csv"]["columns"] == [
        "id_code",
        "diagnosis",
    ]
    assert summary["csv_schemas"]["official_test"]["test.csv"]["columns"] == [
        "id_code"
    ]
    assert summary["class_balance"]["official_train"]["counts_by_grade"] == {
        "0": 1,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 1,
    }
    assert summary["class_balance"]["official_train"]["missing_grades"] == [1, 2, 3]
    assert summary["metadata_comparison"]["mismatching_rows"] == 0
    assert output_dir.joinpath("old_image_metadata.csv").is_file()
    assert output_dir.joinpath("official_train_image_metadata.csv").is_file()
    assert output_dir.joinpath("class_distribution.csv").is_file()
    assert output_dir.joinpath("resolution_distribution_by_grade.csv").is_file()
    assert output_dir.joinpath("audit_summary.json").is_file()
    assert audit_outputs_exist(output_dir)
    output_dir.joinpath("class_distribution.csv").unlink()
    assert audit_outputs_exist(output_dir) is False
    
    official_resolutions = result["resolution_distribution"].query(
        "source == 'official_train' and split == 'all'"
    )
    assert set(official_resolutions["resolution"]) == {"8x8", "16x8"}
    assert set(official_resolutions["diagnosis"]) == {0, 4}
