"""Audit a third-party APTOS copy against the official competition download.

The script takes the root directories of the legacy dataset and official
Kaggle download, then compares the legacy train/validation/test CSV records
and image files with the official labeled ``train.csv`` and ``train_images``.
It also checks the official ``test.csv`` schema and IDs, but never opens or
hashes official test-image files.

The output is a per-image CSV report containing ID, split, label, and
SHA-256 byte-hash comparisons, plus a JSON summary of dataset inventory,
overlap, label agreement, and file identity. 

These results help establish whether the legacy copy matches the official training source before the data
is used in experiments, without exposing test-image contents to the audit.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

# Old APTOS-2019
OLD_PARTITIONS = (
    ("train", "train_1.csv", "train_images/train_images"),
    ("val", "valid.csv", "val_images/val_images"),
    ("test", "test.csv", "test_images/test_images"),
)
# image suffixes
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}

# Areas to report 
REPORT_FIELDS = (
    "id_code",
    "old_split",
    "old_diagnosis",
    "new_diagnosis",
    "label_match",
    "old_sha256",
    "new_sha256",
    "byte_hash_match",
)


def read_csv_records(csv_path, require_diagnosis):
    """Read and validate image IDs and optional diagnosis labels from a CSV.
    Args:
        csv_path: Path to a CSV containing an ``id_code`` column.
        require_diagnosis: Whether a ``diagnosis`` column with grades 0 through
            4 is required and should be parsed.
    Returns:
        A pair of a mapping from normalized image ID to diagnosis (or ``None``
        when labels are not required) and the CSV's column names.
        
    Raises:
        ValueError: If required columns are absent, an ID is empty or
            duplicated, or a required diagnosis is invalid.
            
    This validation creates a reliable ID/label index before comparing the
    differently organized source datasets.
    """
    records = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        if "id_code" not in fields:
            raise ValueError(f"{csv_path} must contain an id_code column")
        if require_diagnosis and "diagnosis" not in fields:
            raise ValueError(f"{csv_path} must contain a diagnosis column")
        
        for row_number, row in enumerate(reader, start=2):
            raw_id = (row.get("id_code") or "").strip()
            image_id = Path(raw_id).stem
            if not image_id:
                raise ValueError(f"{csv_path}:{row_number} has an empty id_code")
            if image_id in records:
                raise ValueError(f"{csv_path} contains duplicate id_code {image_id!r}")
            
            diagnosis = None
            if require_diagnosis:
                try:
                    diagnosis = int(row["diagnosis"])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{csv_path}:{row_number} has an invalid diagnosis"
                    ) from error
                if diagnosis not in range(5):
                    raise ValueError(
                        f"{csv_path}:{row_number} diagnosis must be between 0 and 4"
                    )
            records[image_id] = diagnosis
    return records, fields


def index_images(image_dir):
    """Index supported image files by filename stem.
    Args:
        image_dir: Directory recursively searched for supported image files.
    Returns:
        A mapping from each image ID (the file stem) to its path.
    Raises:
        FileNotFoundError: If ``image_dir`` does not exist as a directory.
        ValueError: If multiple supported files in the tree have the same
            image ID.
            
    Unique IDs are necessary for an unambiguous per-image comparison between
    CSV records and the corresponding image files.
    """
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    images = {}
    for path in image_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image_id = path.stem
        if image_id in images:
            raise ValueError(
                f"Multiple image files share ID {image_id!r} in {image_dir}"
            )
        images[image_id] = path
    return images


def inventory(csv_records, image_paths):
    """Summarize whether a CSV's image IDs agree with the files on disk.
    Args:
        csv_records: Mapping of image IDs listed in a dataset CSV.
        image_paths: Mapping of image IDs found in the image directory.
        
    Returns:
        A dictionary containing CSV and file counts, CSV IDs without files,
        and image IDs without CSV records.
        
    This identifies incomplete or extra source data independently of label
    and byte-content comparisons.
    """
    csv_ids = set(csv_records)
    image_ids = set(image_paths)
    return {
        "csv_rows": len(csv_records),
        "image_files": len(image_paths),
        "csv_ids_missing_images": sorted(csv_ids - image_ids),
        "unlisted_image_ids": sorted(image_ids - csv_ids),
    }


def sha256_file(path):
    """Compute the SHA-256 digest of a file's exact byte contents.
    Args:
        path: Path to the file to read.
        
    Returns:
        The lowercase hexadecimal SHA-256 digest.
        
    Exact-byte hashing determines whether corresponding images are identical
    files, rather than merely sharing an ID or diagnosis.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_sources(old_root, new_root, report_path):
    """Compare a legacy APTOS copy with the official competition data.
    
    Args:
        old_root: Root containing the legacy train, validation, and test
            partitions described by ``OLD_PARTITIONS``.
        new_root: Root containing the official ``train.csv``,
            ``train_images``, and ``test.csv``.
        report_path: Destination for the per-ID CSV comparison report.
        
    Returns:
        A nested summary of source inventories, ID overlap, diagnosis
        agreement, exact-byte hash agreement, official test CSV schema, and
        the report path.
        
    Raises:
        FileNotFoundError: If a required CSV or image directory is missing.
        ValueError: If CSV records or image IDs are invalid or ambiguous.
        
    The comparison checks whether the legacy labeled partitions match the
    official training source while deliberately never opening or hashing
    official test-image files.
    """
    old_labels = {}
    old_splits = {}
    old_paths = {}
    old_partition_summaries = {}
    
    for split, csv_name, image_dir_name in OLD_PARTITIONS:
        csv_path = old_root / csv_name
        records, _ = read_csv_records(csv_path, require_diagnosis=True)
        image_paths = index_images(old_root / image_dir_name)
        old_partition_summaries[split] = inventory(records, image_paths)
        
        for image_id, diagnosis in records.items():
            if image_id in old_labels:
                raise ValueError(
                    f"Old source ID {image_id!r} occurs in multiple CSV partitions"
                )
            old_labels[image_id] = diagnosis
            old_splits[image_id] = split
            old_paths[image_id] = image_paths.get(image_id)
            
    new_labels, _ = read_csv_records(
        new_root / "train.csv", require_diagnosis=True
    )
    new_train_paths = index_images(new_root / "train_images")
    new_test_ids, test_fields = read_csv_records(
        new_root / "test.csv", require_diagnosis=False
    )
    
    old_ids = set(old_labels)
    new_ids = set(new_labels)
    common_ids = old_ids & new_ids
    old_only_ids = old_ids - new_ids
    new_only_ids = new_ids - old_ids
    
    new_train_inventory = inventory(new_labels, new_train_paths)
    test_id_set = set(new_test_ids)
    
    label_matches = []
    label_mismatches = []
    hash_matches = []
    hash_mismatches = []
    comparable_hash_ids = []
    report_rows = []
    
    for image_id in sorted(old_ids | new_ids):
        old_path = old_paths.get(image_id)
        new_path = new_train_paths.get(image_id)
        old_hash = sha256_file(old_path) if old_path is not None else ""
        new_hash = sha256_file(new_path) if new_path is not None else ""
        
        label_match = ""
        if image_id in old_labels and image_id in new_labels:
            label_match = old_labels[image_id] == new_labels[image_id]
            (label_matches if label_match else label_mismatches).append(image_id)
            
        byte_hash_match = ""
        if old_hash and new_hash:
            byte_hash_match = old_hash == new_hash
            comparable_hash_ids.append(image_id)
            (hash_matches if byte_hash_match else hash_mismatches).append(image_id)
            
        report_rows.append(
            {
                "id_code": image_id,
                "old_split": old_splits.get(image_id, ""),
                "old_diagnosis": old_labels.get(image_id, ""),
                "new_diagnosis": new_labels.get(image_id, ""),
                "label_match": label_match,
                "old_sha256": old_hash,
                "new_sha256": new_hash,
                "byte_hash_match": byte_hash_match,
            }
        )
        
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(report_rows)
        
    return {
        "old_source": {
            "csv_rows_by_split": {
                split: details["csv_rows"]
                for split, details in old_partition_summaries.items()
            },
            "unique_ids": len(old_ids),
            "image_inventory_by_split": old_partition_summaries,
        },
        "official_train": {
            "csv_rows": len(new_labels),
            "unique_ids": len(new_ids),
            "image_inventory": new_train_inventory,
        },
        "id_comparison": {
            "common_ids": len(common_ids),
            "old_only_ids": len(old_only_ids),
            "new_train_only_ids": len(new_only_ids),
            "old_only_sample": sorted(old_only_ids)[:20],
            "new_train_only_sample": sorted(new_only_ids)[:20],
        },
        "label_comparison": {
            "comparable_ids": len(label_matches) + len(label_mismatches),
            "matching_labels": len(label_matches),
            "mismatching_labels": len(label_mismatches),
            "mismatching_id_sample": sorted(label_mismatches)[:20],
        },
        "file_hash_comparison": {
            "algorithm": "SHA-256 of exact file bytes",
            "comparable_ids": len(comparable_hash_ids),
            "identical_hashes": len(hash_matches),
            "different_hashes": len(hash_mismatches),
            "different_hash_id_sample": sorted(hash_mismatches)[:20],
        },
        "official_test_csv": {
            "columns": test_fields,
            "rows": len(new_test_ids),
            "has_diagnosis_column": "diagnosis" in test_fields,
            "ids_overlapping_official_train": len(test_id_set & new_ids),
            "test_images_opened_or_hashed": False,
        },
        "per_id_report": str(report_path),
    }


def parse_args():
    """Parse command-line paths for the source comparison.
    
    Returns:
        An ``argparse.Namespace`` with the legacy root, official root, and
        per-ID report destination.
        
    Defaults point to the project's standard local data layout, while command
    line options allow the audit to run against alternate downloads.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare old APTOS IDs, labels, and file hashes with the official "
            "Kaggle training set. Official test images are not read."
        )
    )
    parser.add_argument("--old-root", type=Path, default=Path("data/raw/APTOS-2019"))
    parser.add_argument(
        "--new-root",
        type=Path,
        default=Path("data/raw/APTOS-2019-kaggle-competition"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/aptos_source_comparison.csv"),
        help="Local per-ID report path; keep this file out of Git.",
    )
    return parser.parse_args()


def main():
    """Run the source comparison from command-line arguments.
    
    Writes the per-ID CSV report through :func:`compare_sources` and prints
    the aggregate summary as formatted JSON for command-line users.
    """
    args = parse_args()
    summary = compare_sources(args.old_root, args.new_root, args.report)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
