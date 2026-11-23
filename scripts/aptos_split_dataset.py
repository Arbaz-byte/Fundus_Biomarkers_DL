"""Materialise the APTOS-2019 official split manifest into per-split folders.

Reads:
    configs/splits/aptos_official_split_v1.csv            (id_code, diagnosis, split)
    data/raw/APTOS-2019-kaggle-competition/train.csv      (official labels)
    data/raw/APTOS-2019-kaggle-competition/train_images/  (official train images)

Writes:
    data/processed/APTOS-official-split/
        train_images/  val_images/  test_images/
        train.csv      val.csv      test.csv        (columns: id_code, diagnosis)

Only the official *train* set is used. The Kaggle test set has no public labels.

Usage (from the project root):
    python scripts/aptos_split_dataset.py
    python scripts/aptos_split_dataset.py --mode symlink
    python scripts/aptos_split_dataset.py --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

SPLITS = ("train", "test","val")
VALID_LABELS = {0,1,2,3,4}
IMAGE_EXTS = (".png", ".jpg", ".jpeg")

# Working project dir
PROJECT_ROOT = Path.cwd().resolve()

# Default Directories
DEFAULT_MANIFEST  = PROJECT_ROOT / "configs" / "splits" / "aptos_official_split_v1.csv"

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "APTOS-2019-kaggle-competition"

DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "APTOS-official-split"


# --------------------------------------------------------------------------- 
# Loading + validation
# --------------------------------------------------------------------------- 

def load_manifest(path: Path) -> pd.DataFrame:
    """Load and validate the official APTOS dataset split manifest.

    Args:
        path: Path to a CSV containing ``id_code``, ``diagnosis``, and ``split``
            columns.

    Returns:
        The validated manifest as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        ValueError: If required columns or values are missing or invalid, or if
            an image identifier occurs more than once.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"Split manifest not found : {path}"
                                "Run create_aptos_split.py first")
    
    df = pd.read_csv(path, dtype={"id_code":str})
    expected = {"id_code", "diagnosis", "split"}
    missing_cols = expected - set(df.columns)
    
    # check for missing columns
    if missing_cols:
        raise ValueError(f"Manifest is missing columns : {sorted(missing_cols)}")
    
    # check for Nan values
    if df["id_code"].isna().any() or df["diagnosis"].isna().any() or df["split"].isna().any():
        raise ValueError("Manifest contains empty values, check create_aptos_split.py")
    
    # check for duplicate values
    dupes = df.loc[df["id_code"].duplicated(), "id_code"].tolist()
    if dupes:
        raise ValueError(f"Manifest has duplicate id_codes(leakage risk!) : {dupes[:10]}")
    
    # check for split
    bad_split = sorted(set(df["split"]) - set(SPLITS))
    if bad_split:
        raise ValueError(f"Manifest has unknown split values : {bad_split}")
    
    # check for diagnosis labels 
    df["diagnosis"] = df["diagnosis"].astype(int)
    bad_labels = sorted(set(df["diagnosis"]) - VALID_LABELS)
    if bad_labels:
        raise ValueError(f"Manifest has invalid diagnosis values : {bad_labels}")
    
    return df


def load_official_train(path: Path) -> pd.DataFrame:
    """Load the labeled official APTOS-2019 training CSV.

    Args:
        path: Path to the Kaggle ``train.csv`` file.

    Returns:
        The training metadata containing at least ``id_code`` and ``diagnosis``.

    Raises:
        FileNotFoundError: If the official training CSV is missing.
        ValueError: If either required column is absent.
    """
    
    if not Path(path).is_file():
        raise FileNotFoundError(f"Official train.csv not found at {path}. Make sure the APTOS-2019 dataset has been downloaded and extracted.")
    
    df = pd.read_csv(path, dtype={"id_code": str})
    if not {"id_code", "diagnosis"} <= set(df.columns):
        raise ValueError("Official train.csv must have columns id_code, diagnosis")
    
    return df

def check_manifest_against_official(manifest: pd.DataFrame, official: pd.DataFrame) -> None:
    """Verify manifest identifiers and labels against official training data.

    The manifest must contain only identifiers from the labeled Kaggle training
    set. Diagnostic labels must agree with the official CSV for every manifest
    row. Official training identifiers omitted from the manifest are reported
    as warnings because they are outside the requested split.

    Args:
        manifest: Validated split manifest.
        official: Official APTOS-2019 training metadata.

    Raises:
        ValueError: If a manifest identifier is unknown or a manifest label
            disagrees with the official label.
    """
    
    manifest_ids = set(manifest["id_code"])
    official_ids = set(official["id_code"])
    
    # Unknown values 
    unknown = sorted(manifest_ids - official_ids)
    if unknown:
        raise ValueError(
            f"{len(unknown)} manifest ids are not in the official train.csv"
            f"(e.g. {unknown[:5]}. Test-set images must never enter the split."
        )
    
    # absent values (id_code)
    absent = official_ids - manifest_ids
    if absent:
        print(
            f"[warn] {len(absent)} official train ids are not in the manifest"
            f"and will be ignored.",
            file = sys.stderr,
        )
    
    merged = manifest.merge(official, on="id_code", suffixes=("_manifest", "_official"))
    mismatch = merged[merged["diagnosis_manifest"] != merged["diagnosis_official"].astype(int)]
    if not mismatch.empty:
        raise ValueError(
            f"{len(mismatch)} labels in the manifest disagree with the official train.csv"
            f"(e.g. {mismatch['id_code'].head(5).tolist()})."
        )


def locate_image(image_dir: Path, id_code: str) -> Path | None:
    """Locate an image by identifier in the supported image extensions.

    Args:
        image_dir: Directory containing the source images.
        id_code: Image identifier, normally corresponding to the stem of the
            source filename.

    Returns:
        The matching source path, or ``None`` if no supported image exists.
    """

    for exe in IMAGE_EXTS:
        candidate = Path(image_dir) / f"{id_code}{exe}"
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------------- 
# Materialisation
# --------------------------------------------------------------------------- 

def _place(src: Path, dst: Path, mode: str) -> bool:
    """Place a source image at the destination using the requested mode.

    Existing files or links are reused only when the selected mode already
    represents the correct source. Otherwise, the existing destination is
    replaced.

    Args:
        src: Source image path.
        dst: Destination image path.
        mode: Placement strategy: ``copy``, ``symlink``, or ``hardlink``.

    Returns:
        ``True`` when the destination was written or replaced; ``False`` when
        an existing copy or link was already valid.

    Raises:
        ValueError: If ``mode`` is not supported.
    """
    
    if dst.exists() or dst.is_symlink():
        if mode == "copy" and dst.is_file() and dst.stat().st_size == src.stat().st_size:
            return False # already there
        
        if mode != "copy" and dst.is_symlink() and dst.resolve() == src.resolve():
            return False
        
        dst.unlink()
    
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unknown Mode: {mode}")
    
    return True


def build_split_dataset(manifest_path: Path = DEFAULT_MANIFEST, 
                        raw_dir: Path = DEFAULT_RAW_DIR ,
                        out_dir: Path = DEFAULT_OUT_DIR ,
                        mode: str = "copy", overwrite: bool = False,) ->dict[str, pd.DataFrame]:
    """Build the processed APTOS dataset from the official split manifest.

    Args:
        manifest_path: CSV containing the split assignments.
        raw_dir: Root directory of the extracted Kaggle APTOS dataset.
        out_dir: Destination directory for split folders and metadata CSVs.
        mode: How source images are added to split folders: ``copy``,
            ``symlink``, or ``hardlink``.
        overwrite: Whether to remove existing split image directories before
            rebuilding them.

    Returns:
        A mapping from split name (``train``, ``val``, or ``test``) to a
        DataFrame containing ``id_code`` and ``diagnosis``.

    Raises:
        FileNotFoundError: If required directories or source images are missing.
        RuntimeError: If an output split directory cannot be removed in
            overwrite mode.
        AssertionError: If post-build output verification detects a mismatch.
        ValueError: If manifest validation or cross-source label validation
            fails.
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    train_images_dir = raw_dir / "train_images"
    
    if not train_images_dir.is_dir():
        raise FileNotFoundError(f"Official train_images folder not found : {train_images_dir}")
    
    manifest = load_manifest(manifest_path)
    official = load_official_train(raw_dir/ "train.csv")
    check_manifest_against_official(manifest=manifest, official=official)
    
    # Resolving every source image first so we fail before writing anything.
    sources : dict[str, Path] = {}
    missing : list[str] = []
    
    for id_code in manifest["id_code"]:
        src = locate_image(train_images_dir, id_code)
        if src is None:
            missing.append(id_code)
        else:
            sources[id_code] = src
            
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images from the manifest are missing in {train_images_dir}"
            f"(e.g. {missing[:5]})."
        )
    
    if overwrite:
        for split in SPLITS:
            target = out_dir / f"{split}_images"
            shutil.rmtree(target, ignore_errors=True)
            if target.exists():
                raise RuntimeError(f"Could not fully remove directory: {target}")
    
    result: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        split_df = (
            manifest.loc[manifest["split"] == split, ["id_code", "diagnosis"]].sort_values("id_code")
            .reset_index(drop=True)
        )
        
        img_out = out_dir / f"{split}_images"
        img_out.mkdir(parents=True, exist_ok=True)
        
        written = 0
        for id_code in split_df["id_code"]:
            src = sources[id_code]
            written += _place(src, img_out / src.name, mode)
        
        split_df.to_csv(out_dir / f"{split}.csv", index = False)
        result[split] = split_df
        print(f"[{split:5s}] {len(split_df):5d} images ({written} written) -> {img_out}")
    
    verify_output(out_dir, manifest)
    return result

# --------------------------------------------------------------------------- 
# Post-build verification
# --------------------------------------------------------------------------- 

def verify_output(out_dir: Path, manifest: pd.DataFrame) -> None:
    """Verify that generated folders and CSVs exactly match the manifest.

    Args:
        out_dir: Root directory containing the generated split metadata and
            image folders.
        manifest: Expected split assignments and diagnostic labels.

    Raises:
        AssertionError: If a split CSV or image folder differs from the
            manifest, or if an image identifier appears in more than one split.
    """
    
    out_dir = Path(out_dir)
    seen : dict[str, str] ={}
    
    for split in SPLITS:
        expected_ids = set(manifest.loc[manifest["split"] == split, "id_code"])
        
        csv_df = pd.read_csv(out_dir / f"{split}.csv", dtype={"id_code" : str})
        if set(csv_df['id_code']) != expected_ids or len(csv_df) != len(expected_ids):
            raise AssertionError(f"{split}.csv does not match the manifest.")
        
        files = [p for p in (out_dir / f"{split}_images").iterdir() if not p.name.startswith(".")]
        folder_ids = {p.stem for p in files}
        
        if folder_ids != expected_ids or len(files) != len(expected_ids):
            extra = sorted(folder_ids - expected_ids)[:5]
            gone = sorted(expected_ids - folder_ids)[:5]
            raise AssertionError(
                f"{split}_images does not match the manifest"
                f"(extra: {extra}, missing: {gone}). Re-run with --overwrite"
            )
        
        for id_code in folder_ids:
            if id_code in seen:
                raise AssertionError(f"Leakage! : {id_code} is in both {seen[id_code]} and {split}")
            seen[id_code] = split


def print_summary(result: dict[str, pd.DataFrame]) -> None:
    """Print image counts and diagnosis distributions for each split.

    Args:
        result: Mapping of split names to their split metadata DataFrames.
    """
    print("\nClass distribution (diagnosis 0-4):")
    for split, df in result.items():
        counts = df["diagnosis"].value_counts().reindex(sorted(VALID_LABELS), fill_value=0)
        print(f" {split:5s} n = {len(df):5d} " + " ".join(f"{k}:{v}" for k,v in counts.items()))


def main() -> None:
    """Parse command-line arguments and build the processed APTOS splits.

    The command uses project-relative defaults and prints the resulting split
    sizes and class distributions after the build completes.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the official split manifest CSV.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Root directory of the extracted Kaggle APTOS dataset.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where processed split folders and CSVs are written.",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink", "hardlink"],
        default="copy",
        help="Method used to place source images in the output folders.",
    )
    
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing *_images folders in out_dir and rebuild.")
    
    args = parser.parse_args()
    
    result = build_split_dataset(args.manifest, args.raw_dir, args.out_dir, args.mode, args.overwrite)
    print_summary(result)
    print("\n.....Split completed!")
    


if __name__ == "__main__":
    main()
