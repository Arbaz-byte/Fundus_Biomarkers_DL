"""Create a reproducible APTOS split after duplicate and label-conflict checks."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from skimage.metrics import structural_similarity


DEFAULT_SOURCES = (
    ("train_1.csv", "train_images/train_images"),
    ("valid.csv", "val_images/val_images"),
    ("test.csv", "test_images/test_images"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge labelled APTOS CSVs, remove exact duplicate images, group "
            "near-duplicates, and write one stratified split."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/APTOS-2019"))
    parser.add_argument(
        "--source",
        action="append",
        metavar="CSV=IMAGE_DIR",
        help=(
            "Source CSV and its image directory, relative to --data-root. "
            "Repeat for each source. Defaults to train_1.csv, valid.csv, test.csv."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("configs/splits/aptos_v1.csv"))
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
        help="Number of folds assigned to the sealed test split.",
    )
    parser.add_argument(
        "--val-folds",
        type=int,
        default=3,
        help="Number of folds assigned to validation; remaining folds are train.",
    )
    parser.add_argument(
        "--diagnoses",
        nargs="+",
        type=int,
        help="Optional diagnosis labels to include, for example --diagnoses 0 2 4.",
    )
    parser.add_argument(
        "--phash-max-distance",
        type=int,
        default=4,
        help="Maximum pHash Hamming distance for candidate near-duplicate pairs.",
    )
    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=0.95,
        help="Minimum SSIM required to group a pHash candidate as a near-duplicate.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=256,
        help="Square image side length used for SSIM comparisons.",
    )
    parser.add_argument(
        "--conflicting-labels",
        choices=("exclude", "keep"),
        default="exclude",
        help="Exclude conflicting duplicate groups or retain them in one split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    return parser.parse_args()


def read_sources(args):
    sources = args.source or DEFAULT_SOURCES
    records = []
    seen_ids = set()
    diagnoses = set(args.diagnoses) if args.diagnoses is not None else None

    for source in sources:
        if isinstance(source, str):
            if "=" not in source:
                raise ValueError(f"Invalid --source {source!r}; expected CSV=IMAGE_DIR")
            csv_name, image_dir = source.split("=", 1)
        else:
            csv_name, image_dir = source

        frame = pd.read_csv(args.data_root / csv_name)
        if not {"id_code", "diagnosis"}.issubset(frame.columns):
            raise ValueError(f"{csv_name} must contain id_code and diagnosis columns")

        for record in frame.itertuples(index=False):
            image_id = str(record.id_code)
            if image_id in seen_ids:
                raise ValueError(f"Duplicate id_code across source CSVs: {image_id}")
            seen_ids.add(image_id)
            diagnosis = int(record.diagnosis)
            if diagnoses is not None and diagnosis not in diagnoses:
                continue

            image_path = args.data_root / image_dir / f"{image_id}.png"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                pixel_hash = hashlib.sha256(
                    rgb.tobytes() + str(rgb.size).encode("ascii")
                ).hexdigest()
                perceptual_hash = int(str(imagehash.phash(rgb)), 16)
                small_rgb = np.asarray(rgb.resize((args.resize, args.resize))).copy()
            records.append(
                {
                    "id_code": image_id,
                    "diagnosis": diagnosis,
                    "pixel_hash": pixel_hash,
                    "phash": perceptual_hash,
                    "small_rgb": small_rgb,
                }
            )

    if not records:
        raise ValueError("No labelled images remain after applying the diagnosis filter")
    return records


def group_duplicates(records, max_phash_distance, ssim_threshold):
    parent = list(range(len(records)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    pixel_groups = defaultdict(list)
    for index, record in enumerate(records):
        pixel_groups[record["pixel_hash"]].append(index)
    for indexes in pixel_groups.values():
        for index in indexes[1:]:
            union(indexes[0], index)

    for index, record in enumerate(records):
        for other_index in range(index):
            other = records[other_index]
            if record["pixel_hash"] == other["pixel_hash"]:
                continue
            distance = (record["phash"] ^ other["phash"]).bit_count()
            if distance > max_phash_distance:
                continue
            similarity = structural_similarity(
                record["small_rgb"],
                other["small_rgb"],
                channel_axis=2,
                data_range=255,
            )
            if similarity >= ssim_threshold:
                union(index, other_index)

    components = defaultdict(list)
    for index in range(len(records)):
        components[find(index)].append(index)
    return components


def make_split(records, components, args):
    conflicting_components = {
        group_id
        for group_id, indexes in components.items()
        if len({records[index]["diagnosis"] for index in indexes}) > 1
    }

    retained = []
    excluded_rows = 0
    for group_id, indexes in components.items():
        if group_id in conflicting_components and args.conflicting_labels == "exclude":
            excluded_rows += len(indexes)
            continue

        if group_id in conflicting_components:
            representatives = indexes
        else:
            by_pixel_hash = {}
            for index in indexes:
                record = records[index]
                current = by_pixel_hash.get(record["pixel_hash"])
                if current is None or record["id_code"] < records[current]["id_code"]:
                    by_pixel_hash[record["pixel_hash"]] = index
            representatives = list(by_pixel_hash.values())

        for index in representatives:
            retained.append(
                {
                    "id_code": records[index]["id_code"],
                    "diagnosis": records[index]["diagnosis"],
                    "group": group_id,
                }
            )

    frame = pd.DataFrame(retained).sort_values("id_code").reset_index(drop=True)
    if frame.empty:
        raise ValueError("No images remain after duplicate handling")
    if frame["id_code"].duplicated().any():
        raise AssertionError("The retained id_code values are not unique")
    if args.n_splits < 2 or args.test_folds < 1 or args.val_folds < 1:
        raise ValueError("n-splits must be at least 2 and test/validation folds must be positive")
    if args.test_folds + args.val_folds >= args.n_splits:
        raise ValueError("test-folds + val-folds must be less than n-splits")

    splitter = StratifiedGroupKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )
    fold = np.full(len(frame), -1, dtype=int)
    for fold_index, (_, held_out) in enumerate(
        splitter.split(np.zeros(len(frame)), frame["diagnosis"], groups=frame["group"])
    ):
        fold[held_out] = fold_index
    if (fold < 0).any():
        raise AssertionError("Some retained rows were not assigned to a fold")

    frame["split"] = np.where(
        fold < args.test_folds,
        "test",
        np.where(fold < args.test_folds + args.val_folds, "val", "train"),
    )
    if frame.groupby("group")["split"].nunique().gt(1).any():
        raise AssertionError("A duplicate group crosses split boundaries")
    return frame[["id_code", "diagnosis", "split"]], len(conflicting_components), excluded_rows


def main():
    args = parse_args()
    if not 0 <= args.phash_max_distance <= 64:
        raise ValueError("phash-max-distance must be between 0 and 64")
    if not 0 <= args.ssim_threshold <= 1:
        raise ValueError("ssim-threshold must be between 0 and 1")
    if args.resize < 1:
        raise ValueError("resize must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {args.output}; choose a new --output or pass --overwrite"
        )

    records = read_sources(args)
    components = group_duplicates(
        records,
        max_phash_distance=args.phash_max_distance,
        ssim_threshold=args.ssim_threshold,
    )
    result, conflict_count, excluded_rows = make_split(records, components, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print(f"Wrote {args.output}: {len(result)} rows")
    print(f"Conflicting duplicate groups: {conflict_count}")
    print(f"Source rows excluded for conflicting labels: {excluded_rows}")
    print("Split counts:")
    print(result["split"].value_counts().reindex(["train", "val", "test"]).to_string())
    print("Diagnosis counts by split:")
    print(pd.crosstab(result["split"], result["diagnosis"]).reindex(["train", "val", "test"]).to_string())


if __name__ == "__main__":
    main()
