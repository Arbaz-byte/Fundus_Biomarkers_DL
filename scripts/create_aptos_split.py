"""Create a reproducible, leakage-resistant split from official APTOS training data.

The script reads the official labeled ``train.csv`` and matching
``train_images/`` directory, checks their ID inventory, groups likely
near-duplicates with perceptual hash and SSIM, then assigns groups to
diagnosis-stratified train/validation/local-test partitions. Defaults target
approximately 70%/15%/15% using a fixed seed.

The unlabeled competition test set is never read. Diagnosis-conflicting
duplicate groups are retained and reported rather than silently dropped.
Because APTOS provides no patient IDs and duplicate groups may not divide
evenly, this split is not patient-level and realized proportions can vary.
The generated manifest contains ``id_code``, ``diagnosis`` and ``split`` and
should remain local and git-ignored.
"""

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence, TypedDict, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from fundus.duplicates import MIN_SSIM_SIZE, build_duplicate_groups

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_ORDER = ["train", "val", "test"]


class SplitRecord(TypedDict):
    """Fields required to assign a labeled image to a dataset partition."""

    id_code: str
    diagnosis: int


class TrainingRecord(SplitRecord):
    """A labeled training record resolved to its official image file."""

    image_path: Path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse paths and options for creating the local split manifest.

    Args:
        argv: Optional command-line tokens. When omitted, arguments are read
            from the process command line.

    Returns:
        Parsed options for the official data root, duplicate detection,
        splitting, and output manifest.

    This function only configures the command-line interface; it does not read
    data or perform the split.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create diagnosis-stratified, duplicate-group-aware train/val/test "
            "partitions from the official labeled APTOS training set."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/APTOS-2019-kaggle-competition"),
        help="Root of the official APTOS competition download.",
    )
    parser.add_argument("--train-csv", default="train.csv")
    parser.add_argument("--image-dir", default="train_images")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/aptos_official_split_v1.csv"),
        help="Local split manifest path; keep this per-image file out of Git.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-splits",
        type=int,
        default=20,
        help="Number of stratified group folds used to approximate split proportions.",
    )
    parser.add_argument(
        "--test-folds",
        type=int,
        default=3,
        help="Number of folds assigned to the sealed local test partition "
        "(3/20 = 15%%).",
    )
    parser.add_argument(
        "--val-folds",
        type=int,
        default=3,
        help="Number of folds assigned to validation (3/20 = 15%%); "
        "the remaining folds (14/20 = 70%%) go to train.",
    )
    # Duplicate-group-aware settings: near-duplicates (same photo, different
    # processing) must land in the same split to prevent leakage.
    parser.add_argument(
        "--phash-max-distance",
        type=int,
        default=4,
        help="Max perceptual-hash Hamming distance for a pair to be a duplicate candidate.",
    )
    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=0.95,
        help="Minimum SSIM for a hash-matching pair to be linked.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=128,
        help="Square side length used for perceptual-hash and SSIM comparison.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing manifest.",
    )
    return parser.parse_args(argv)


def read_training_records(
    train_csv: Union[str, Path],
    image_dir: Union[str, Path],
) -> list[TrainingRecord]:
    """Read official training labels and resolve each ID to exactly one image.

    Args:
        train_csv: Official labeled training CSV containing ``id_code`` and
            ``diagnosis`` columns.
        image_dir: Directory containing the official training images.

    Returns:
        Records sorted by image ID. Each record contains ``id_code``,
        ``diagnosis``, and the corresponding ``image_path``. Stable ordering
        supports reproducible duplicate grouping and split assignment.

    Raises:
        FileNotFoundError: If the CSV or image directory does not exist.
        ValueError: If the schema, IDs, diagnoses, image IDs, or CSV/image
            inventory are invalid or inconsistent.
    """
    train_csv = Path(train_csv)
    image_dir = Path(image_dir)

    if not train_csv.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Training image directory not found: {image_dir}")

    records_by_id = {}
    with train_csv.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        missing_columns = {"id_code", "diagnosis"} - set(columns)
        if missing_columns:
            raise ValueError(
                f"{train_csv} is missing required columns: "
                f"{', '.join(sorted(missing_columns))}"
            )

        for row_number, row in enumerate(reader, start=2):
            raw_id = (row.get("id_code") or "").strip()
            image_id = Path(raw_id).stem
            if not image_id:
                raise ValueError(f"{train_csv}:{row_number} has an empty id_code")
            if image_id in records_by_id:
                raise ValueError(
                    f"{train_csv}:{row_number} has duplicate id_code {image_id!r}"
                )

            try:
                diagnosis = int(row["diagnosis"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{train_csv}:{row_number} has an invalid diagnosis"
                ) from error
            if diagnosis not in range(5):
                raise ValueError(
                    f"{train_csv}:{row_number} diagnosis must be between 0 and 4"
                )

            records_by_id[image_id] = {"id_code": image_id, "diagnosis": diagnosis}

    images_by_id = {}
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image_id = path.stem
        if image_id in images_by_id:
            raise ValueError(
                f"Multiple training images share ID {image_id!r} in {image_dir}"
            )
        images_by_id[image_id] = path

    csv_ids = set(records_by_id)
    image_ids = set(images_by_id)
    missing_images = sorted(csv_ids - image_ids)
    unlisted_images = sorted(image_ids - csv_ids)

    if missing_images or unlisted_images:
        details = []
        if missing_images:
            details.append(f"CSV IDs without images: {missing_images[:10]}")
        if unlisted_images:
            details.append(f"Images without CSV rows: {unlisted_images[:10]}")
        raise ValueError("Training CSV/image inventory mismatch; " + "; ".join(details))

    return [
        {**records_by_id[image_id], "image_path": images_by_id[image_id]}
        for image_id in sorted(records_by_id)
    ]


def create_manifest(
    records: Sequence[SplitRecord],
    group_ids: Union[Sequence[object], np.ndarray],
    *,
    n_splits: int = 20,
    test_folds: int = 3,
    val_folds: int = 3,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Assign records to approximate 70/15/15 partitions without group leakage.

    Args:
        records: Labeled image records to partition.
        group_ids: Duplicate-group identifier for each record, in matching
            order. Every member of a group is assigned to the same partition.
        n_splits: Number of stratified group folds used to approximate the
            proportions.
        test_folds: Number of folds reserved for the local held-out test.
        val_folds: Number of folds reserved for validation.
        seed: Random seed controlling fold assignment.

    Returns:
        A tuple containing the per-image manifest, per-image group/split
        assignments used for validation, and the number of duplicate groups
        containing conflicting diagnoses.

    Raises:
        ValueError: If record/group lengths, fold counts, IDs, or minimum
            per-grade group counts are invalid.
        RuntimeError: If the splitter leaves rows unassigned or divides a
            duplicate group across partitions.

    Conflicting diagnoses inside a duplicate group are retained and reported;
    records are never silently removed or relabeled.
    """
    if len(records) != len(group_ids):
        raise ValueError("There must be one duplicate-group ID per training record")
    if n_splits < 2 or test_folds < 1 or val_folds < 1:
        raise ValueError("n-splits must be at least 2 and test/val folds must be positive")
    if test_folds + val_folds >= n_splits:
        raise ValueError("test-folds + val-folds must be less than n-splits")

    frame = pd.DataFrame(
        {
            "id_code": [record["id_code"] for record in records],
            "diagnosis": [record["diagnosis"] for record in records],
            "group_id": [str(group_id) for group_id in group_ids],
        }
    )
    if frame["id_code"].duplicated().any():
        raise ValueError("Training record IDs must be unique")

    group_counts_by_grade = (
        frame.groupby("diagnosis")["group_id"].nunique().reindex(range(5), fill_value=0)
    )
    insufficient = group_counts_by_grade[group_counts_by_grade < n_splits]
    if not insufficient.empty:
        details = ", ".join(
            f"grade {grade}: {count} groups" for grade, count in insufficient.items()
        )
        raise ValueError(
            f"Each grade needs at least n-splits={n_splits} distinct groups "
            f"for this stratification; {details}. Reduce --n-splits."
        )

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_by_row = np.full(len(frame), -1, dtype=int)
    dummy_features = np.zeros((len(frame), 1), dtype=np.uint8)

    for fold_index, (_, held_out_rows) in enumerate(
        splitter.split(dummy_features, frame["diagnosis"], groups=frame["group_id"])
    ):
        fold_by_row[held_out_rows] = fold_index

    if (fold_by_row < 0).any():
        raise RuntimeError("The splitter did not assign every training row")

    frame["split"] = np.where(
        fold_by_row < test_folds,
        "test",
        np.where(fold_by_row < test_folds + val_folds, "val", "train"),
    )

    if frame.groupby("group_id")["split"].nunique().gt(1).any():
        raise RuntimeError("A duplicate group was assigned to multiple splits")

    conflicting_groups = frame.groupby("group_id")["diagnosis"].nunique().gt(1).sum()

    manifest = (
        frame[["id_code", "diagnosis", "split"]]
        .sort_values("id_code")
        .reset_index(drop=True)
    )
    group_assignments = frame[["id_code", "group_id", "split"]]
    return manifest, group_assignments, int(conflicting_groups)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Build, save, and summarize the official-training split manifest.

    Args:
        argv: Optional command-line tokens, useful when invoking the entry
            point programmatically. If omitted, process arguments are parsed.

    The manifest contains one row per labeled official training image and is
    written to the configured local output path. A summary reports actual
    split sizes, diagnosis counts, duplicate links, and conflicting groups.

    Raises:
        ValueError: If duplicate thresholds or split options are invalid, or
            if the training files are malformed or inconsistent.
        FileExistsError: If the output already exists and ``--overwrite`` was
            not supplied.
    """
    args = parse_args(argv)

    if not 0 <= args.phash_max_distance <= 64:
        raise ValueError("--phash-max-distance must be between 0 and 64")
    if not 0 <= args.ssim_threshold <= 1:
        raise ValueError("--ssim-threshold must be between 0 and 1")
    if args.resize < MIN_SSIM_SIZE:
        raise ValueError(f"--resize must be at least {MIN_SSIM_SIZE}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {args.output}; choose another path or pass --overwrite"
        )

    train_csv = args.data_root / args.train_csv
    image_dir = args.data_root / args.image_dir
    records = read_training_records(train_csv, image_dir)
    image_paths = [record["image_path"] for record in records]

    group_ids, duplicate_links = build_duplicate_groups(
        image_paths,
        max_hash_distance=args.phash_max_distance,
        min_ssim=args.ssim_threshold,
        ssim_size=args.resize,
    )
    manifest, group_assignments, conflicting_groups = create_manifest(
        records,
        group_ids,
        n_splits=args.n_splits,
        test_folds=args.test_folds,
        val_folds=args.val_folds,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)

    row_counts = manifest["split"].value_counts().reindex(SPLIT_ORDER, fill_value=0)
    print(f"Wrote {args.output}: {len(manifest)} rows")
    print(f"Duplicate groups: {group_assignments['group_id'].nunique()}")
    print(f"Near-duplicate links: {len(duplicate_links)}")
    print(f"Duplicate groups with conflicting diagnoses: {conflicting_groups}")
    print("Rows by split:")
    print(row_counts.to_string())
    print("Approximate row proportions:")
    print((row_counts / len(manifest)).round(4).to_string())
    print("Diagnosis counts by split:")
    print(
        pd.crosstab(manifest["split"], manifest["diagnosis"])
        .reindex(index=SPLIT_ORDER, columns=range(5), fill_value=0)
        .to_string()
    )
    print("Duplicate groups by split:")
    print(
        group_assignments.groupby("split")["group_id"]
        .nunique()
        .reindex(SPLIT_ORDER, fill_value=0)
        .to_string()
    )


if __name__ == "__main__":
    main()
