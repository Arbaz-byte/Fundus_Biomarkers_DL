"""Test the official APTOS train/validation/local-test split workflow.

Synthetic labeled training images and CSV rows exercise input validation,
reproducibility, approximate split proportions, duplicate-group isolation,
conflicting-label retention, and overwrite protection. An intentionally
unreadable image in the official test directory verifies that only labeled
training files are processed.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts.create_aptos_split import create_manifest, main, read_training_records


def _smooth_image(rng: np.random.Generator, size: int = 32) -> np.ndarray:
    """Create a deterministic textured RGB image for the temporary dataset.

    Args:
        rng: NumPy random generator used to create the low-resolution texture.
        size: Width and height of the square output image.

    Returns:
        A ``size`` by ``size`` RGB array of unsigned 8-bit values.

    The smoothed texture produces reproducible, non-mirrored images that can
    also be re-encoded as JPEG for duplicate-detection tests.
    """
    small = rng.integers(0, 256, size=(4, 4, 3), dtype=np.uint8)
    return np.asarray(Image.fromarray(small).resize((size, size), Image.BICUBIC))


def _write_training_fixture(root: Path) -> None:
    """Create temporary official-style training files and a sealed-test sentinel.

    Args:
        root: Temporary dataset root where ``train.csv``, ``train_images/``,
            and an intentionally unreadable ``test_images/`` file are created.

    The training fixture includes exact, JPEG-reencoded, and mirrored variants
    of one image, including a duplicate with a conflicting diagnosis. The
    invalid test image ensures accidental access to the competition test
    directory fails loudly.
    """
    image_dir = root / "train_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(42)
    first_image_pixels = None
    
    # Produces 50 images. For the very first one (grade0_00.png), 
    # it re-reads the PNG bytes so they can later be written verbatim to another file —
    # creating a true byte-for-byte duplicate that perceptual-hash / SSIM logic should flag.
    for grade in range(5):
        for index in range(10):
            image_id = f"grade{grade}_{index:02d}"
            pixels = _smooth_image(rng)
            Image.fromarray(pixels).save(image_dir / f"{image_id}.png")
            rows.append({"id_code": image_id, "diagnosis": grade})
            if grade == 0 and index == 0:
                first_image_pixels = pixels  # anchor for the duplicate variants

    assert first_image_pixels is not None, "fixture logic changed: no anchor image"

    # Variant 1: exact byte duplicate with a conflicting label.
    # The split builder should retain both records, flag the group, and keep
    # them together.
    Image.fromarray(first_image_pixels).save(image_dir / "conflicting_copy.png")
    rows.append({"id_code": "conflicting_copy", "diagnosis": 1})

    # Variant 2: lossy JPEG re-encode — same visual content, different bytes.
    # quality=95 (not 30) because the anchor is 32x32 random noise, which is
    # the worst case for lossy compression; a low quality would drop SSIM
    # below the grouping threshold and break the positive test.
    Image.fromarray(first_image_pixels).save(
        image_dir / "reencoded_copy.jpg", format="JPEG", quality=95
    )
    rows.append({"id_code": "reencoded_copy", "diagnosis": 0})  
    
    
    # Variant 3: horizontal mirror — visually a different sample, so the
    # split builder should NOT fold it into grade0_00's group.
    Image.fromarray(np.fliplr(first_image_pixels)).save(image_dir / "mirrored_copy.png")
    rows.append({"id_code": "mirrored_copy", "diagnosis": 0})

    with (root / "train.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id_code", "diagnosis"])
        writer.writeheader()
        writer.writerows(rows)

    # The real test_images/ is unlabeled and must never be read. If production
    # code globbed under data_root, PIL would choke on this file and fail loudly.
    test_dir = root / "test_images"
    test_dir.mkdir()
    (test_dir / "must_not_be_opened.png").write_bytes(b"not an image!")


def _run_arguments(root: Path, output_path: Path) -> list[str]:
    """Build command-line arguments for a small, reproducible test split.

    Args:
        root: Temporary official-data root.
        output_path: Destination for the generated split manifest.

    Returns:
        CLI tokens using five folds and a fixed seed to keep the test fast and
        deterministic.
    """
    return [
        "--data-root", str(root),
        "--output", str(output_path),
        "--n-splits", "5",
        "--test-folds", "1",
        "--val-folds", "1",
        "--seed", "17",
        "--resize", "32",
    ]


def test_main(tmp_path: Path) -> None:
    """Check reproducibility, duplicate isolation, conflicts, and test-set handling.

    Runs the script twice with the same seed and compares manifests. It also
    verifies all labeled rows remain, duplicate variants share a partition,
    conflicting labels are preserved, and the unlabeled test image is ignored.
    """
    _write_training_fixture(tmp_path)

    first_path = tmp_path / "processed" / "first.csv"
    second_path = tmp_path / "processed" / "second.csv"

    main(_run_arguments(tmp_path, first_path))
    main(_run_arguments(tmp_path, second_path))

    first = pd.read_csv(first_path)
    second = pd.read_csv(second_path)

    # --- repeatability -------------------------------------------------------
    pd.testing.assert_frame_equal(first, second)  # repeatability
    
    # --- row count: 50 base + conflicting_copy + reencoded_copy + mirrored ---
    assert len(first) == 53  # 50 base + 3 variant
    
    # --- all three partitions are populated (n_splits=5, test=1, val=1) ------
    assert set(first["split"]) == {"train", "val", "test"}

    split_of = first.set_index("id_code")["split"]
    
    # --- duplicate isolation: exact duplicate + lossy re-encode stay together
    assert split_of["grade0_00"] == split_of["conflicting_copy"]
    assert split_of["grade0_00"] == split_of["reencoded_copy"]

    # --- retained conflicts: the conflicting row kept its (wrong) label ------
    # If this fails, the builder silently dropped or relabelled the duplicate.
    labels = first.set_index("id_code")["diagnosis"]
    assert labels["grade0_00"] == 0
    assert labels["conflicting_copy"] == 1  # conflicting label retained

    # --- official test set ignored: the poisoned file never entered manifest -
    assert "must_not_be_opened" not in first["id_code"].values


def test_main_refuses_to_overwrite(tmp_path: Path) -> None:
    """Require an explicit overwrite flag before replacing an existing manifest."""
    _write_training_fixture(tmp_path)
    out = tmp_path / "processed" / "m.csv"
    
    main(_run_arguments(tmp_path, out))
    
    
    with pytest.raises(FileExistsError):
        main(_run_arguments(tmp_path, out))
    main(_run_arguments(tmp_path, out) + ["--overwrite"])


def test_read_training_records_rejects_missing_image(tmp_path: Path) -> None:
    """Reject a training CSV whose image directory is missing a referenced ID."""
    _write_training_fixture(tmp_path)
    (tmp_path / "train_images" / "grade4_09.png").unlink()
    
    with pytest.raises(ValueError) as excinfo:
        read_training_records(tmp_path / "train.csv", tmp_path / "train_images")
        
    msg = str(excinfo.value)
    assert "inventory mismatch" in msg.lower()
    assert "grade4_09" in msg


def test_default_folds_give_70_15_15() -> None:
    """Approximate the intended proportions and keep every duplicate group intact.

    Variable-sized groups make exact percentages impossible; this fixture
    checks that the 14/3/3 fold defaults remain close to 70/15/15 while no
    group's rows are assigned to multiple partitions.
    """
    rng = np.random.default_rng(0)
    records, group_ids = [], []
    group = 0
    for grade, n_groups in enumerate([400, 100, 250, 60, 90]):
        for _ in range(n_groups):
            size = int(rng.choice([1, 1, 1, 2]))  # some duplicate groups of 2
            for _ in range(size):
                records.append({"id_code": f"r{len(records):05d}", "diagnosis": grade})
                group_ids.append(group)
            group += 1
            
    manifest, assignments, conflicts = create_manifest(records, group_ids, seed=42)
    
    shares = manifest["split"].value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.70, abs=0.03)
    assert shares["val"] == pytest.approx(0.15, abs=0.03)
    assert shares["test"] == pytest.approx(0.15, abs=0.03)
    assert assignments.groupby("group_id")["split"].nunique().max() == 1
    assert conflicts == 0


def test_create_manifest_rejects_bad_fold_config() -> None:
    """Reject a fold configuration that leaves no folds for training."""
    records = [{"id_code": "a", "diagnosis": 0}]
    with pytest.raises(ValueError):
        create_manifest(records, [0], n_splits=5, test_folds=3, val_folds=2)


