"""Test duplicate detection behavior used to prevent split leakage.

The tests verify that exact copies and a JPEG re-encoding are grouped, while a
mirrored image and unrelated image remain separate. They also cover transitive
group formation, empty and single-image inputs, and validation of the minimum
image size required for SSIM.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fundus.duplicates import build_duplicate_groups


def _noise(seed: int, size: int = 32) -> np.ndarray:
    """Create a deterministic textured RGB image for duplicate tests.

    Args:
        seed: Random seed that determines the image pattern.
        size: Width and height of the square output image.

    Returns:
        A ``size`` by ``size`` RGB array of unsigned 8-bit pixel values.

    Resizing a small random pattern creates texture that remains similar after
    JPEG encoding while different seeds produce distinct test images.
    """
    small = np.random.default_rng(seed).integers(0, 256, (4, 4, 3), dtype=np.uint8)
    return np.asarray(Image.fromarray(small).resize((size, size), Image.BICUBIC))


def test_groups_exact_reencoded_and_ignores_mirror_and_others(tmp_path: Path) -> None:
    """Group identical and JPEG-reencoded images, but not mirrors or unrelated images.

    This checks both positive matches and important non-matches so duplicate
    grouping does not merge images merely because they are related by a
    transformation or have broadly similar content.
    """
    base = _noise(1)
    paths = {
        "a": tmp_path / "a.png",
        "a_copy": tmp_path / "a_copy.png",
        "a_jpg": tmp_path / "a_jpg.jpg",
        "a_mirror": tmp_path / "a_mirror.png",
        "other": tmp_path / "other.png",
    }
    Image.fromarray(base).save(paths["a"])
    Image.fromarray(base).save(paths["a_copy"])
    Image.fromarray(base).save(paths["a_jpg"], format="JPEG", quality=95)
    Image.fromarray(np.fliplr(base)).save(paths["a_mirror"])
    Image.fromarray(_noise(2)).save(paths["other"])

    order = list(paths)
    groups, links = build_duplicate_groups([paths[k] for k in order], ssim_size=32)
    gid = dict(zip(order, groups))

    assert gid["a"] == gid["a_copy"] == gid["a_jpg"]
    assert gid["a_mirror"] != gid["a"]
    assert gid["other"] != gid["a"]
    assert all(i > j for i, j, _ in links)


def test_transitive_grouping(tmp_path: Path) -> None:
    """Merge a connected chain of matching images into one duplicate group."""
    base = _noise(3)
    for name in "abc":
        Image.fromarray(base).save(tmp_path / f"{name}.png")
    groups, links = build_duplicate_groups(sorted(tmp_path.glob("*.png")), ssim_size=32)
    assert len(set(groups)) == 1
    assert len(links) == 3


def test_empty_and_single_input(tmp_path: Path) -> None:
    """Return valid empty and singleton results without inventing duplicate links."""
    groups, links = build_duplicate_groups([])
    assert len(groups) == 0 and links == []
    Image.fromarray(_noise(4)).save(tmp_path / "x.png")
    groups, links = build_duplicate_groups([tmp_path / "x.png"], ssim_size=32)
    assert list(groups) == [0] and links == []


def test_rejects_tiny_ssim_size(tmp_path: Path) -> None:
    """Reject SSIM image sizes smaller than the metric's supported window."""
    with pytest.raises(ValueError):
        build_duplicate_groups([], ssim_size=4)
