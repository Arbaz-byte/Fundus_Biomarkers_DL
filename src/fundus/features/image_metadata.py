"""
Image-level metadata for quality and acquisition-variation analysis.

Two groups of measurements are returned:

* Whole-image measurements (``width`` ... ``black_border_fraction``). These keep the original
  definitions: statistics are computed over *all* pixels, including the black border, and the
  Laplacian is computed at native resolution.
  
* Field-of-view (FOV) measurements (``fov_*``). These are computed only inside the detected
  retinal disc, on a copy resized to a fixed size, so they are not mechanically tied to the amount
  of black border or to the native pixel count.

All measurements are descriptive proxies; none is a clinical quality score.
"""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterable, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

ImagePath = Union[str, Path]

FOV_REFERENCE_SIZE = 512  # longest side, in pixels, used for all fov_* measurements
_NAN = float("nan")

FOV_COLUMNS = [
    "fov_area_fraction",
    "fov_bbox_width_fraction",
    "fov_bbox_height_fraction",
    "fov_center_offset_x",
    "fov_center_offset_y",
    "fov_brightness_mean",
    "fov_red_mean",
    "fov_green_mean",
    "fov_blue_mean",
    "fov_red_std",
    "fov_green_std",
    "fov_blue_std",
    "fov_blur_laplacian_variance",
]


def estimate_fov_mask(rgb: np.ndarray, black_threshold: int = 10) -> np.ndarray:
    """Return a boolean mask of the retinal disc (field of view) for an RGB uint8 image.

    Pixels with any channel above ``black_threshold`` are candidates. A morphological opening removes
    isolated bright specks, a closing fills small holes (dark vessels and lesions), and only the largest
    connected component is kept.
    """
    candidate = (rgb.max(axis=2) > black_threshold).astype(np.uint8)
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, small_kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, large_kernel)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if component_count <= 1:
        return np.zeros(rgb.shape[:2], dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def fov_metadata(rgb: np.ndarray, black_threshold: int = 10,
                 reference_size: int = FOV_REFERENCE_SIZE) -> dict:
    """Measure the retinal disc and colour/sharpness statistics inside it.

    The image is first resized so that its longest side equals ``reference_size``. Doing this makes the
    sharpness proxy comparable between images of different native resolution.
    """
    height, width, _ = rgb.shape
    scale = reference_size / max(height, width)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    small = cv2.resize(rgb, new_size, interpolation=interpolation)
    small_height, small_width, _ = small.shape

    mask = estimate_fov_mask(small, black_threshold)
    if not mask.any():
        return {column: _NAN for column in FOV_COLUMNS}

    rows, columns = np.nonzero(mask)
    bbox_height = rows.max() - rows.min() + 1
    bbox_width = columns.max() - columns.min() + 1
    pixels = small[mask].astype(np.float32)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    # Erode the mask so the sharp edge of the retinal disc does not dominate the Laplacian.
    inner = cv2.erode(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
    if not inner.any():
        inner = mask
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    return {
        "fov_area_fraction": float(mask.mean()),
        "fov_bbox_width_fraction": float(bbox_width / small_width),
        "fov_bbox_height_fraction": float(bbox_height / small_height),
        "fov_center_offset_x": float((columns.mean() - (small_width - 1) / 2) / small_width),
        "fov_center_offset_y": float((rows.mean() - (small_height - 1) / 2) / small_height),
        "fov_brightness_mean": float(gray[mask].mean()),
        "fov_red_mean": float(pixels[:, 0].mean()),
        "fov_green_mean": float(pixels[:, 1].mean()),
        "fov_blue_mean": float(pixels[:, 2].mean()),
        "fov_red_std": float(pixels[:, 0].std()),
        "fov_green_std": float(pixels[:, 1].std()),
        "fov_blue_std": float(pixels[:, 2].std()),
        "fov_blur_laplacian_variance": float(laplacian[inner].var()),
    }


def image_metadata(image_path: ImagePath, black_threshold: int = 10, include_fov: bool = True) -> dict:
    """Measure basic image properties and image-quality proxies.

    ``black_border_fraction`` is the fraction of all pixels whose three RGB channels are at or below
    ``black_threshold``. It estimates dark border area; it is not a segmentation of the retinal field of
    view. Set ``include_fov=False`` to get only the original whole-image measurements.
    """
    path = Path(image_path)
    if not 0 <= black_threshold <= 255:
        raise ValueError("black_threshold must be between 0 and 255")

    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Unable to read image: {path}") from error

    height, width, _ = rgb.shape
    pixels = rgb.astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    dark_pixels = np.all(rgb <= black_threshold, axis=2)

    record = {
        "image_id": path.stem,
        "image_path": str(path),
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
        "file_size_bytes": path.stat().st_size,
        "brightness_mean": float(gray.mean()),
        "red_mean": float(pixels[:, :, 0].mean()),
        "green_mean": float(pixels[:, :, 1].mean()),
        "blue_mean": float(pixels[:, :, 2].mean()),
        "red_std": float(pixels[:, :, 0].std()),
        "green_std": float(pixels[:, :, 1].std()),
        "blue_std": float(pixels[:, :, 2].std()),
        "blur_laplacian_variance": float(laplacian.var()),
        "black_border_fraction": float(dark_pixels.mean()),
    }
    if include_fov:
        record.update(fov_metadata(rgb, black_threshold))
    return record


def _safe_metadata(image_path, black_threshold, include_fov):
    """Worker used by build_image_metadata; returns (record, error_message)."""
    try:
        return image_metadata(image_path, black_threshold, include_fov), None
    except ValueError as error:
        return None, str(error)


def build_image_metadata(
    image_paths: Iterable[ImagePath],
    black_threshold: int = 10,
    include_fov: bool = True,
    on_error: str = "raise",
    n_jobs: int = 1,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Build one metadata row per image path, preserving input order.

    ``on_error="skip"`` drops unreadable images and lists them in ``DataFrame.attrs["failed"]``;
    the default ``"raise"`` stops at the first unreadable image. ``n_jobs > 1`` uses worker processes.
    """
    if on_error not in {"raise", "skip"}:
        raise ValueError("on_error must be 'raise' or 'skip'")
    paths = list(image_paths)
    worker = partial(_safe_metadata, black_threshold=black_threshold, include_fov=include_fov)

    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            results = pool.map(worker, paths, chunksize=8)
            results = _with_progress(results, len(paths), show_progress)
            outcomes = list(results)
    else:
        outcomes = list(_with_progress(map(worker, paths), len(paths), show_progress))

    records, failed = [], []
    for path, (record, error) in zip(paths, outcomes):
        if record is not None:
            records.append(record)
        elif on_error == "raise":
            raise ValueError(error)
        else:
            failed.append({"image_path": str(path), "error": error})

    frame = pd.DataFrame(records)
    frame.attrs["failed"] = failed
    return frame


def _with_progress(iterator, total, show_progress):
    if not show_progress:
        return iterator
    try:
        from tqdm.auto import tqdm
    except ImportError:  # progress bars are optional
        return iterator
    return tqdm(iterator, total=total, desc="Profiling images")
