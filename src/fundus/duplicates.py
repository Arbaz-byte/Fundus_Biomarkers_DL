"""
Detect and group near-duplicate fundus images.

The grouping helper identifies likely copies so they can be kept together
during dataset splitting. Otherwise, a copy in training and its near-duplicate
in evaluation could make performance appear better than it is on unseen
images.

"""
from typing import List, Sequence, Tuple, Union
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def _fingerprint(path, ssim_size: int):
    """
    Load an image and create the representations used for comparison.
    Args:
        path: Filesystem path to an image readable by Pillow.
        ssim_size: Side length to which the RGB image is resized for both
            perceptual-hash and structural-similarity comparisons.
    Returns:
        A pair containing the perceptual hash as an integer and a copied
        ``(ssim_size, ssim_size, 3)`` RGB pixel array.
        
    Raises:
        FileNotFoundError: If the image path does not exist.
        OSError: If Pillow cannot open or decode the image.
    """
    with Image.open(path) as image:
        small = image.convert("RGB").resize((ssim_size, ssim_size))

    return int(str(imagehash.phash(small)), 16), np.asarray(small).copy()


def build_duplicate_groups( image_paths: Sequence[Union[str, Path]], max_hash_distance: int = 4, min_ssim: float = 0.95, ssim_size: int = 128,
) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    
    """
    Group images that are likely to be copies or near-duplicates.

    Each image is resized to ``ssim_size`` square pixels and compared with
    earlier images. 
    
    A pair is linked only if its perceptual-hash Hamming distance is at most ``max_hash_distance`` and its structural similarity
    (SSIM) is at least ``min_ssim``. Linked pairs are merged transitively: if A matches B and B matches C, all three receive the same group ID.

    Keeping each resulting group within one dataset split helps prevent duplicate-image leakage from inflating evaluation performance. Since
    comparisons use the hash as a pre-filter, cropped, rotated, or heavily recolored copies may be missed. This implementation compares every image
    pair, so the pairwise comparison work grows quadratically with the number of input images.

    Args:
        image_paths: Ordered sequence of image paths. Input positions are
            preserved in both returned values.
        max_hash_distance: Maximum allowed perceptual-hash Hamming distance
            for a pair to receive the more expensive SSIM comparison.
            Defaults to 4.
        min_ssim: Minimum SSIM required to link a pair after the hash check.
            Defaults to 0.95.
        ssim_size: Side length in pixels used to resize each image before
            comparison. Defaults to 128.

    Returns:
        A pair ``(group_ids, links)``:

        - ``group_ids`` is a NumPy integer array with one entry per input
        image. Equal IDs indicate images in the same connected component;
        the specific ID values are implementation details.
        
        - ``links`` is a list of ``(i, j, ssim)`` tuples for pairs that passed
        both thresholds, where ``i`` and ``j`` are indices into
        ``image_paths`` and ``ssim`` is their similarity score.

    Raises:
        FileNotFoundError: If an image path does not exist.
        OSError: If an image cannot be opened or decoded.
    """
    records = [_fingerprint(path, ssim_size) for path in image_paths]
    parent = list(range(len(records)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    links = []

    for index, (hash_value, pixels) in enumerate(records):
        for other in range(index):
            other_hash, other_pixels = records[other]
            if (hash_value ^ other_hash).bit_count() > max_hash_distance:
                continue
            similarity = structural_similarity(pixels, other_pixels, channel_axis=2, data_range=255)
            if similarity >= min_ssim:
                links.append((index, other, float(similarity)))
                root_a, root_b = find(index), find(other)
                if root_a != root_b:
                    parent[root_b] = root_a


    return np.array([find(index) for index in range(len(records))]), links
