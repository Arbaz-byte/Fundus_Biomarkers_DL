"""Audit the old APTOS copy against the official competition download.

The official competition test images are deliberately excluded. Only the
official training images and the old copy's three labeled partitions are
profiled; per-image outputs are written under the git-ignored data/processed
directory.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from fundus.features.image_metadata import build_image_metadata

if __package__:
    from .compare_aptos_sources import OLD_PARTITIONS, index_images, read_csv_records
else:
    from compare_aptos_sources import OLD_PARTITIONS, index_images, read_csv_records


REPORT_COLUMNS = {
    "id_code",
    "old_split",
    "old_diagnosis",
    "new_diagnosis",
    "label_match",
    "old_sha256",
    "new_sha256",
    "byte_hash_match",
}
AUDIT_OUTPUT_FILENAMES = (
    "old_image_metadata.csv",
    "official_train_image_metadata.csv",
    "class_distribution.csv",
    "resolution_distribution_by_grade.csv",
    "audit_summary.json",
)
GRADES = range(5)
METADATA_EXCLUDED_COMPARISON_COLUMNS = {
    "diagnosis",
    "image_path",
    "source",
    "split",
}


def audit_outputs_exist(output_dir):
    """Check whether an audit output directory contains the complete report set.
    Args:
        output_dir: Directory expected to contain all files listed in
            ``AUDIT_OUTPUT_FILENAMES``.
    Returns:
        ``True`` only when every expected output exists as a file.
        
    This lightweight check lets callers determine whether an audit has already
    produced its full set of outputs without opening or parsing them.
    """
    output_dir = Path(output_dir)
    return all((output_dir / filename).is_file() for filename in AUDIT_OUTPUT_FILENAMES)


def _read_schema(csv_path, require_diagnosis):
    """Read a dataset CSV and describe its input schema.
    
    Args:
        csv_path: Path to the CSV file to read.
        require_diagnosis: Whether the shared CSV reader should require a
            diagnosis column.
            
    Returns:
        A pair containing the parsed image records and a schema summary with
        the CSV column names and row count.
        
    The audit records schemas alongside its results so missing labels and
    differences between the old and official data sources are visible.
    """
    records, columns = read_csv_records(csv_path, require_diagnosis)
    return records, {
        "columns": columns,
        "rows": len(records),
    }


def _bool_value(value):
    """Parse a serialized boolean while preserving unknown or empty values.
    
    Args:
        value: A boolean or text representation of ``true`` or ``false``.
        
    Returns:
        ``True`` or ``False`` for recognized values, or ``None`` when the
        value cannot be interpreted as a boolean.
        
    Comparison reports are read from CSV, where boolean values may be stored
    as strings; retaining unrecognized values as ``None`` allows the audit to
    report incomplete comparisons rather than treating them as matches.
    """
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _profile_images(image_paths, labels, splits, source, n_jobs, include_fov, show_progress):
    """Extract metadata for one source and attach its labels and split context.
    
    Args:
        image_paths: Mapping from image IDs to their filesystem paths.
        labels: Mapping from image IDs to integer diagnoses, when available.
        splits: Mapping from image IDs to split names.
        source: Source identifier stored in the returned metadata.
        n_jobs: Number of workers for image metadata extraction.
        include_fov: Whether to include field-of-view measurements.
        show_progress: Whether to display extraction progress when supported.

    Returns:
        A dataframe of image metadata with ``id_code``, ``diagnosis``,
        ``split``, and ``source`` columns added.
        
    Raises:
        ValueError: If ``image_paths`` is empty.
        
    This shared step gives both sources a consistent per-image profile for
    later distribution summaries and metadata comparison.
    """
    if not image_paths:
        raise ValueError(f"No images found for source {source!r}")
    
    image_ids = sorted(image_paths)
    metadata = build_image_metadata(
        [image_paths[image_id] for image_id in image_ids],
        include_fov=include_fov,
        n_jobs=n_jobs,
        show_progress=show_progress,
    ).rename(columns={"image_id": "id_code"})
    metadata["diagnosis"] = pd.array(
        [labels.get(image_id) for image_id in metadata["id_code"]],
        dtype="Int64",
    )
    metadata["split"] = [
        splits.get(image_id, "unmatched") for image_id in metadata["id_code"]
    ]
    metadata["source"] = source
    return metadata


def _inventory(csv_records, image_paths):
    """Count CSV/image inventory mismatches for a dataset partition.
    
    Args:
        csv_records: Mapping of image IDs present in the partition CSV.
        image_paths: Mapping of image IDs found in the corresponding image
            directory.
            
    Returns:
        Counts for CSV rows, image files, CSV IDs without image files, and
        image files without CSV rows.
        
    These counts expose incomplete or inconsistent source downloads without
    requiring the audit consumer to compare the identifiers manually.
    """
    csv_ids = set(csv_records)
    image_ids = set(image_paths)
    return {
        "csv_rows": len(csv_records),
        "image_files": len(image_paths),
        "csv_ids_missing_images": len(csv_ids - image_ids),
        "images_without_csv_rows": len(image_ids - csv_ids),
    }


def _distribution_tables(metadata):
    """Build diagnosis-count and image-resolution distribution tables.
    
    Args:
        metadata: Combined per-image metadata dataframe for the sources being
            audited; rows without a diagnosis are excluded.
            
    Returns:
        A pair of dataframes: diagnosis counts and fractions by source/split,
        and counts grouped by source, split, diagnosis, width, and height.
        
    The aggregate tables make class imbalance and resolution patterns easy
    to inspect without exposing additional per-image details.
    """
    labeled = metadata.dropna(subset=["diagnosis"]).copy()
    labeled["diagnosis"] = labeled["diagnosis"].astype(int)
    
    class_rows = []
    for source, source_frame in labeled.groupby("source", sort=True):
        split_frames = [("all", source_frame)]
        split_frames.extend(
            (split, frame)
            for split, frame in source_frame.groupby("split", sort=True)
            if split != "all"
        )
        for split, frame in split_frames:
            counts = frame["diagnosis"].value_counts().reindex(GRADES, fill_value=0)
            total = int(counts.sum())
            for grade, count in counts.items():
                class_rows.append(
                    {
                        "source": source,
                        "split": split,
                        "diagnosis": int(grade),
                        "count": int(count),
                        "fraction": float(count / total) if total else 0.0,
                    }
                )
    class_distribution = pd.DataFrame(class_rows)
    
    resolution_frames = []
    for source, source_frame in labeled.groupby("source", sort=True):
        frames = [("all", source_frame)]
        frames.extend(
            (split, frame)
            for split, frame in source_frame.groupby("split", sort=True)
            if split != "all"
        )
        for split, frame in frames:
            counts = (
                frame.groupby(["diagnosis", "width", "height"], as_index=False)
                .size()
                .rename(columns={"size": "count"})
            )
            counts.insert(0, "split", split)
            counts.insert(0, "source", source)
            counts["resolution"] = (
                counts["width"].astype(str) + "x" + counts["height"].astype(str)
            )
            resolution_frames.append(counts)
    resolution_distribution = pd.concat(resolution_frames, ignore_index=True)
    return class_distribution, resolution_distribution


def _metadata_comparison(old_metadata, official_metadata):
    """Compare shared metadata fields for image IDs present in both sources.
    
    Args:
        old_metadata: Per-image metadata extracted from the superseded copy.
        official_metadata: Per-image metadata extracted from official
            training images.
            
    Returns:
        A summary containing the number of compared IDs, matching and
        mismatching rows, and names of the compared columns.
        
    Diagnosis, source, split, and image path are excluded because they are not
    comparable image-derived measurements across these two datasets.
    """
    comparable_columns = sorted(
        (set(old_metadata.columns) & set(official_metadata.columns))
        - METADATA_EXCLUDED_COMPARISON_COLUMNS
        - {"id_code"}
    )
    merged = old_metadata[["id_code", *comparable_columns]].merge(
        official_metadata[["id_code", *comparable_columns]],
        on="id_code",
        how="inner",
        suffixes=("_old", "_official"),
        validate="one_to_one",
    )
    row_matches = pd.Series(True, index=merged.index)
    for column in comparable_columns:
        old_values = merged[f"{column}_old"]
        official_values = merged[f"{column}_official"]
        row_matches &= old_values.eq(official_values) | (
            old_values.isna() & official_values.isna()
        )
    return {
        "compared_ids": len(merged),
        "matching_rows": int(row_matches.sum()),
        "mismatching_rows": int((~row_matches).sum()),
        "compared_columns": comparable_columns,
    }


def run_dataset_audit( old_root, official_root, comparison_report, output_dir, n_jobs=1,
    include_fov=False, show_progress=False, ):
    """Profile both labeled sources and write local per-image and aggregate tables.

    Args:
        old_root: Root of the superseded third-party copy.
        official_root: Root of the official Kaggle competition download.
        comparison_report: Per-ID report created by ``compare_aptos_sources.py``.
        output_dir: Local output directory, normally under ``data/processed``.
        n_jobs: Number of workers for image metadata extraction.
        include_fov: Whether to calculate the additional FOV measurements.
        show_progress: Whether to show progress bars when ``tqdm`` is installed.

    Returns:
        A dictionary containing an aggregate ``summary`` and the metadata and
        distribution dataframes used by the notebook.

    Raises:
        FileNotFoundError: If an input CSV, image directory, or report is missing.
        ValueError: If an input schema or generated comparison report is invalid.
    """
    old_root = Path(old_root)
    official_root = Path(official_root)
    comparison_report = Path(comparison_report)
    output_dir = Path(output_dir)
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1")
    if not comparison_report.is_file():
        raise FileNotFoundError(f"Comparison report not found: {comparison_report}")
    
    report = pd.read_csv(
        comparison_report,
        dtype={"id_code": "string"},
        keep_default_na=False,
    )
    missing_columns = REPORT_COLUMNS - set(report.columns)
    if missing_columns:
        raise ValueError(
            "Comparison report is missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    report["id_code"] = report["id_code"].str.strip()
    if report["id_code"].eq("").any() or report["id_code"].duplicated().any():
        raise ValueError("Comparison report must contain unique, non-empty id_code values")
    for column in REPORT_COLUMNS - {"id_code"}:
        report[column] = report[column].astype(str)
        
    old_csvs = {}
    old_paths = {}
    old_labels = {}
    old_splits = {}
    old_inventory = {}
    old_source_csv_labels = {}
    for split, csv_name, image_dir_name in OLD_PARTITIONS:
        records, schema = _read_schema(old_root / csv_name, require_diagnosis=True)
        paths = index_images(old_root / image_dir_name)
        old_csvs[split] = schema
        old_inventory[split] = _inventory(records, paths)
        old_source_csv_labels.update(records)
        for image_id, image_path in paths.items():
            if image_id in old_paths:
                raise ValueError(
                    f"Old source image ID {image_id!r} occurs in multiple partitions"
                )
            old_paths[image_id] = image_path
            old_labels[image_id] = records.get(image_id)
            old_splits[image_id] = split
            
    official_labels, official_train_schema = _read_schema(
        official_root / "train.csv",
        require_diagnosis=True,
    )
    official_paths = index_images(official_root / "train_images")
    official_inventory = _inventory(official_labels, official_paths)

    official_test_records, official_test_schema = _read_schema(
        official_root / "test.csv",
        require_diagnosis=False,
    )
    
    report_old = report.loc[report["old_split"].str.strip().ne("")]
    report_official = report.loc[report["new_diagnosis"].str.strip().ne("")]
    old_report_ids = set(report_old["id_code"])
    official_report_ids = set(report_official["id_code"])

    old_report_labels = {
        row.id_code: int(row.old_diagnosis)
        for row in report_old.itertuples(index=False)
    }
    official_report_labels = {
        row.id_code: int(row.new_diagnosis)
        for row in report_official.itertuples(index=False)
    }
    common_report = report.loc[
        report["old_split"].str.strip().ne("")
        & report["new_diagnosis"].str.strip().ne("")
    ]
    label_status = [_bool_value(value) for value in common_report["label_match"]]
    hash_rows = report.loc[
        report["old_sha256"].str.strip().ne("")
        & report["new_sha256"].str.strip().ne("")
    ]
    hash_status = [_bool_value(value) for value in hash_rows["byte_hash_match"]]

    old_report_label_mismatches = sum(
        old_source_csv_labels.get(image_id) != diagnosis
        for image_id, diagnosis in old_report_labels.items()
    )
    official_report_label_mismatches = sum(
        official_labels.get(image_id) != diagnosis
        for image_id, diagnosis in official_report_labels.items()
    )

    old_metadata = _profile_images(
        old_paths,
        old_labels,
        old_splits,
        "old_copy",
        n_jobs,
        include_fov,
        show_progress,
    )
    official_metadata = _profile_images(
        official_paths,
        official_labels,
        {image_id: "train" for image_id in official_paths},
        "official_train",
        n_jobs,
        include_fov,
        show_progress,
    )
    class_distribution, resolution_distribution = _distribution_tables(
        pd.concat([old_metadata, official_metadata], ignore_index=True)
    )
    metadata_comparison = _metadata_comparison(old_metadata, official_metadata)

    class_balance = {}
    for source in class_distribution["source"].unique():
        source_counts = class_distribution.loc[
            (class_distribution["source"] == source)
            & (class_distribution["split"] == "all")
        ].set_index("diagnosis")["count"]
        positive_counts = source_counts[source_counts > 0]
        class_balance[source] = {
            "counts_by_grade": {
                str(grade): int(source_counts.get(grade, 0)) for grade in GRADES
            },
            "missing_grades": [
                int(grade) for grade in GRADES if source_counts.get(grade, 0) == 0
            ],
            "largest_to_smallest_present_class_ratio": (
                float(positive_counts.max() / positive_counts.min())
                if not positive_counts.empty
                else None
            ),
        }

    inventory_issue_count = sum(
        item["csv_ids_missing_images"] + item["images_without_csv_rows"]
        for item in old_inventory.values()
    ) + official_inventory["csv_ids_missing_images"] + official_inventory[
        "images_without_csv_rows"
    ]
    report_id_issue_count = (
        len(old_report_ids.symmetric_difference(set(old_source_csv_labels)))
        + len(official_report_ids.symmetric_difference(set(official_labels)))
    )
    label_mismatch_count = sum(value is False for value in label_status)
    hash_mismatch_count = sum(value is False for value in hash_status)
    report_consistency_issue_count = (
        old_report_label_mismatches
        + official_report_label_mismatches
        + sum(value is None for value in label_status)
        + sum(value is None for value in hash_status)
    )

    summary = {
        "audit_complete": (
            inventory_issue_count == 0
            and report_id_issue_count == 0
            and report_consistency_issue_count == 0
            and label_mismatch_count == 0
            and hash_mismatch_count == 0
            and metadata_comparison["mismatching_rows"] == 0
        ),
        "csv_schemas": {
            "old_copy": old_csvs,
            "official_train": {"train.csv": official_train_schema},
            "official_test": {"test.csv": official_test_schema},
        },
        "image_inventory": {
            "old_copy_by_split": old_inventory,
            "official_train": official_inventory,
            "old_copy_profiled_images": len(old_metadata),
            "official_train_profiled_images": len(official_metadata),
        },
        "source_comparison_report": {
            "old_report_ids": len(old_report_ids),
            "official_train_report_ids": len(official_report_ids),
            "common_ids": len(old_report_ids & official_report_ids),
            "old_only_ids": len(old_report_ids - official_report_ids),
            "official_train_only_ids": len(official_report_ids - old_report_ids),
            "matching_labels": sum(value is True for value in label_status),
            "mismatching_labels": label_mismatch_count,
            "identical_byte_hashes": sum(value is True for value in hash_status),
            "different_byte_hashes": hash_mismatch_count,
            "old_report_label_mismatches_against_old_csv": old_report_label_mismatches,
            "official_report_label_mismatches_against_train_csv": (
                official_report_label_mismatches
            ),
        },
        "metadata_comparison": metadata_comparison,
        "class_balance": class_balance,
        "resolution_summary": {
            source: {
                "distinct_resolution_pairs": int(
                    resolution_distribution.loc[
                        (resolution_distribution["source"] == source)
                        & (resolution_distribution["split"] == "all"),
                        "resolution",
                    ].nunique()
                )
            }
            for source in resolution_distribution["source"].unique()
        },
        "official_test_handling": {
            "csv_columns": official_test_schema["columns"],
            "csv_rows": official_test_schema["rows"],
            "has_diagnosis_column": (
                "diagnosis" in official_test_schema["columns"]
            ),
            "test_images_opened_or_hashed": False,
        },
        "output_files": {
            "old_image_metadata": "old_image_metadata.csv",
            "official_train_image_metadata": "official_train_image_metadata.csv",
            "class_distribution": "class_distribution.csv",
            "resolution_distribution_by_grade": "resolution_distribution_by_grade.csv",
            "summary": "audit_summary.json",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    old_metadata.to_csv(output_dir / "old_image_metadata.csv", index=False)
    official_metadata.to_csv(
        output_dir / "official_train_image_metadata.csv",
        index=False,
    )
    class_distribution.to_csv(output_dir / "class_distribution.csv", index=False)
    resolution_distribution.to_csv(
        output_dir / "resolution_distribution_by_grade.csv",
        index=False,
    )
    with (output_dir / "audit_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)

    return {
        "summary": summary,
        "old_metadata": old_metadata,
        "official_metadata": official_metadata,
        "class_distribution": class_distribution,
        "resolution_distribution": resolution_distribution,
    }


def parse_args():
    """Parse command-line options for running the dataset audit.
    
    Returns:
        An ``argparse.Namespace`` containing the old and official dataset
        roots, comparison report, output directory, and metadata options.
        
    Keeping the defaults here provides a reproducible project-local command
    while still allowing paths and processing options to be overridden.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Audit the old APTOS copy and official competition training images. "
            "Official test-image files are never opened."
        )
    )
    parser.add_argument("--old-root", type=Path, default=Path("data/raw/APTOS-2019"))
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path("data/raw/APTOS-2019-kaggle-competition"),
    )
    parser.add_argument(
        "--comparison-report",
        type=Path,
        default=Path("data/processed/aptos_source_comparison.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/aptos_dataset_audit"),
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--include-fov",
        action="store_true",
        help="Also calculate field-of-view metadata; disabled by default for intake.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Display progress bars when tqdm is installed.",
    )
    return parser.parse_args()


def main():
    """Run the audit from command-line arguments and print its aggregate summary.

    The detailed per-image and distribution outputs are written by
    :func:`run_dataset_audit`; printing only the summary gives command-line
    users a concise completion report.
    """
    args = parse_args()
    result = run_dataset_audit(
        args.old_root,
        args.official_root,
        args.comparison_report,
        args.output_dir,
        n_jobs=args.n_jobs,
        include_fov=args.include_fov,
        show_progress=args.show_progress,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
