"""Detect and group near-duplicate fundus images.

The grouping helper identifies likely copies so they can be kept together
during dataset splitting. Otherwise, a copy in training and its near-duplicate
in evaluation could make performance appear better than it is on unseen
images.
"""
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import imagehash
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

# structural_similarity's default window is 7x7, so images must be >= 7 px.
MIN_SSIM_SIZE = 7


def _fingerprint(path, ssim_size: int):
    """Load an image and create the representations used for comparison.

    Args:
        path: Filesystem path to an image readable by Pillow.
        ssim_size: Side length to which the RGB image is resized for both
            perceptual-hash and structural-similarity comparisons.

    Returns:
        A pair containing the 64-bit perceptual hash as a Python int and a
        copied ``(ssim_size, ssim_size, 3)`` uint8 RGB pixel array.

    Raises:
        FileNotFoundError: If the image path does not exist.
        OSError: If Pillow cannot open or decode the image.
    """
    with Image.open(path) as image:
        small = image.convert("RGB").resize((ssim_size, ssim_size))

    return int(str(imagehash.phash(small)), 16), np.asarray(small).copy()


def _hamming_to_all(hashes: np.ndarray, value: int) -> np.ndarray:
    """Hamming distance between ``value`` and every entry of a uint64 array."""
    xor = hashes ^ np.uint64(value)
    return np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def build_duplicate_groups(
    image_paths: Sequence[Union[str, Path]],
    max_hash_distance: int = 4,
    min_ssim: float = 0.95,
    ssim_size: int = 128,
) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    """Group images that are likely to be copies or near-duplicates.

    Each image is resized to ``ssim_size`` square pixels and compared with
    earlier images. A pair is linked only if its perceptual-hash Hamming
    distance is at most ``max_hash_distance`` and its SSIM is at least
    ``min_ssim``. Linked pairs are merged transitively: if A matches B and B
    matches C, all three receive the same group ID.

    Because the hash is used as a pre-filter, cropped, rotated, mirrored or
    heavily recolored copies may be missed. Every image pair is still
    considered, so work grows quadratically with the number of images (the
    hash pre-filter is vectorised; only hash-matching pairs pay for SSIM).

    Args:
        image_paths: Ordered sequence of image paths. Input positions are
            preserved in both returned values.
        max_hash_distance: Maximum perceptual-hash Hamming distance (0-64)
            for a pair to receive the SSIM comparison. Defaults to 4.
        min_ssim: Minimum SSIM required to link a pair. Defaults to 0.95.
        ssim_size: Side length in pixels used to resize each image before
            comparison (>= 7). Defaults to 128.

    Returns:
        ``(group_ids, links)``:

        - ``group_ids``: NumPy integer array, one entry per input image.
        Equal IDs mean the same connected component; the ID values
        themselves are an implementation detail.
        - ``links``: list of ``(i, j, ssim)`` for pairs that passed both
        thresholds, with ``i > j`` indices into ``image_paths``.

    Raises:
        ValueError: If ``ssim_size`` is smaller than 7.
        FileNotFoundError: If an image path does not exist.
        OSError: If an image cannot be opened or decoded.
    """
    if ssim_size < MIN_SSIM_SIZE:
        raise ValueError(f"ssim_size must be at least {MIN_SSIM_SIZE}")

    records = [_fingerprint(path, ssim_size) for path in image_paths]
    hashes = np.array([hash_value for hash_value, _ in records], dtype=np.uint64)
    parent = list(range(len(records)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    links = []

    for index, (hash_value, pixels) in enumerate(records):
        if index == 0:
            continue
        distances = _hamming_to_all(hashes[:index], hash_value)
        for other in np.flatnonzero(distances <= max_hash_distance):
            other = int(other)
            similarity = structural_similarity(
                pixels, records[other][1], channel_axis=2, data_range=255
            )
            if similarity >= min_ssim:
                links.append((index, other, float(similarity)))
                root_a, root_b = find(index), find(other)
                if root_a != root_b:
                    parent[root_b] = root_a

    return np.array([find(index) for index in range(len(records))], dtype=int), links
