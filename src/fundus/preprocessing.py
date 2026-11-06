"""Deterministic preprocessing for fundus photographs.

Baseline pipeline: ``crop_to_fov`` -> ``pad_to_square`` -> ``resize_square`` (see ``preprocess_basic``).
The contrast variants (``local_average_subtraction``, ``clahe_l_channel``) are separate functions and are
**not** applied by default; they are compared later under a leak-free protocol.

All functions take and return RGB ``uint8`` arrays of shape (height, width, 3). None of them uses a label.
"""
from typing import Optional, Tuple

import cv2
import numpy as np

from fundus.features.image_metadata import estimate_fov_mask

MASK_REFERENCE_SIZE = 512  # the retina mask is estimated on a copy whose longest side is this many pixels


def _check_rgb(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 array of shape (height, width, 3)")


def _resize_for_mask(rgb: np.ndarray, reference_size: int) -> Tuple[np.ndarray, float]:
    height, width = rgb.shape[:2]
    scale = reference_size / max(height, width)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(rgb, new_size, interpolation=interpolation), scale


def fov_bounding_box(
    rgb: np.ndarray, black_threshold: int = 10, reference_size: int = MASK_REFERENCE_SIZE
) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box of the retinal disc as ``(top, bottom, left, right)`` in full-resolution pixels.

    ``bottom`` and ``right`` are exclusive, so ``rgb[top:bottom, left:right]`` is the crop. The mask is
    estimated on a copy resized to ``reference_size`` (the morphology kernels are sized for that scale) and
    the box is rounded outward when mapped back. Returns ``None`` if no retina is found.
    """
    _check_rgb(rgb)
    height, width = rgb.shape[:2]
    small, scale = _resize_for_mask(rgb, reference_size)
    mask = estimate_fov_mask(small, black_threshold)
    if not mask.any():
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    top = max(0, int(np.floor(rows[0] / scale)))
    bottom = min(height, int(np.ceil((rows[-1] + 1) / scale)))
    left = max(0, int(np.floor(columns[0] / scale)))
    right = min(width, int(np.ceil((columns[-1] + 1) / scale)))
    return top, bottom, left, right


def crop_to_fov(rgb: np.ndarray, black_threshold: int = 10) -> np.ndarray:
    """Remove the black border around the retinal disc. If no retina is found the image is returned unchanged."""
    box = fov_bounding_box(rgb, black_threshold)
    if box is None:
        return rgb
    top, bottom, left, right = box
    return rgb[top:bottom, left:right]


def pad_to_square(rgb: np.ndarray, fill: int = 0) -> np.ndarray:
    """Pad the shorter side symmetrically with ``fill`` so the image becomes square (no stretching)."""
    _check_rgb(rgb)
    height, width = rgb.shape[:2]
    if height == width:
        return rgb
    side = max(height, width)
    pad_top = (side - height) // 2
    pad_bottom = side - height - pad_top
    pad_left = (side - width) // 2
    pad_right = side - width - pad_left
    return cv2.copyMakeBorder(rgb, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(fill, fill, fill))


def resize_square(rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize a square image to ``size`` x ``size`` (area interpolation when shrinking)."""
    _check_rgb(rgb)
    if rgb.shape[0] != rgb.shape[1]:
        raise ValueError("resize_square expects a square image; call pad_to_square first")
    if size <= 0:
        raise ValueError("size must be positive")
    if rgb.shape[0] == size:
        return rgb
    interpolation = cv2.INTER_AREA if size < rgb.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(rgb, (size, size), interpolation=interpolation)


def preprocess_basic(rgb: np.ndarray, size: int = 256, black_threshold: int = 10) -> np.ndarray:
    """Baseline pipeline: crop to the retinal disc, pad to a square, resize to ``size`` x ``size``."""
    return resize_square(pad_to_square(crop_to_fov(rgb, black_threshold)), size)


def local_average_subtraction(
    rgb: np.ndarray,
    sigma_fraction: float = 1 / 30,
    mask: Optional[np.ndarray] = None,
    black_threshold: int = 10,
) -> np.ndarray:
    """Subtract a local average to reduce illumination differences (Graham-style), respecting the retina mask.

    ``out = 4 * image - 4 * local_average + 128`` inside the retinal disc, saturated to [0, 255]; pixels outside
    the disc are set to 0. The local average is a *mask-normalised* Gaussian blur (blur of ``image * mask``
    divided by blur of ``mask``), so the black border does not leak into it. A plain blur would give a bright rim
    with a dark ring around it near the edge and turn the border grey.

    ``sigma`` is ``sigma_fraction`` of the image width, so apply this **after** resizing to a fixed size for a
    comparable effect across images. If ``mask`` is not given it is estimated with ``estimate_fov_mask``; if no
    retina is found the whole image is treated as retina.
    """
    _check_rgb(rgb)
    if mask is None:
        mask = estimate_fov_mask(rgb, black_threshold)
    if not mask.any():
        mask = np.ones(rgb.shape[:2], dtype=bool)
    mask_float = mask.astype(np.float32)
    sigma = max(1.0, rgb.shape[1] * sigma_fraction)
    image_float = rgb.astype(np.float32) * mask_float[..., None]
    weighted_sum = cv2.GaussianBlur(image_float, (0, 0), sigma)
    weight = cv2.GaussianBlur(mask_float, (0, 0), sigma)[..., None]
    local_average = weighted_sum / np.maximum(weight, 1e-6)
    out = (4.0 * image_float - 4.0 * local_average * mask_float[..., None] + 128.0 * mask_float[..., None])
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def clahe_l_channel(rgb: np.ndarray, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on the lightness channel only (colour is kept)."""
    _check_rgb(rgb)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
