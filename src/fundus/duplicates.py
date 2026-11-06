"""Near-duplicate image detection (used to keep copies of one photograph in the same split)."""
from typing import List, Sequence, Tuple, Union
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def _fingerprint(path, ssim_size: int):
    with Image.open(path) as image:
        small = image.convert("RGB").resize((ssim_size, ssim_size))
    return int(str(imagehash.phash(small)), 16), np.asarray(small).copy()


def build_duplicate_groups(
    image_paths: Sequence[Union[str, Path]],
    max_hash_distance: int = 4,
    min_ssim: float = 0.95,
    ssim_size: int = 128,
) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    """Group near-duplicate images.

    Two images are linked when their perceptual hashes differ by at most ``max_hash_distance`` bits and the
    structural similarity (SSIM) of their ``ssim_size`` x ``ssim_size`` versions is at least ``min_ssim``.
    Links are merged transitively. Returns ``(group_ids, links)`` where ``group_ids[i]`` is the group of
    ``image_paths[i]`` and ``links`` lists ``(i, j, ssim)``.

    Limitation: the hash pre-filter can miss copies that were cropped, rotated or heavily recoloured.
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
