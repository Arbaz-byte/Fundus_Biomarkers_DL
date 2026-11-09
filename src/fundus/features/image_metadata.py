"""Image-level metadata for quality and acquisition-variation analysis.

Two groups of measurements are returned:

* Whole-image measurements (``width`` ... ``black_border_fraction``). These keep the original
  definitions: statistics are computed over *all* pixels, including the black border, and the
  Laplacian is computed at native resolution.
  
* Field-of-view (FOV) measurements (``fov_*``). These are computed only inside the detected
  retinal disc, on a copy resized to a fixed size, so they are not mechanically tied to the amount
  of black border or to the native pixel count.

These are descriptive image proxies, not clinical quality scores or
diagnoses.
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
    """Estimate the retinal field-of-view mask from an RGB image.
    
    Pixels with any channel greater than ``black_threshold`` are candidates.
    Morphological opening removes isolated bright specks, closing fills small
    dark holes, and only the largest connected component is retained. This
    mask isolates the main retinal region for downstream FOV measurements.
    
    Args:
        rgb: RGB image array with shape ``(height, width, 3)``.
        black_threshold: Candidate-pixel threshold, applied independently to
            each pixel's maximum color-channel value. Defaults to 10.
            
    Returns:
        Boolean array with shape ``(height, width)``; ``True`` indicates a
        pixel in the estimated retinal field. Returns an all-false mask if no
        foreground component is detected.
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


def fov_metadata(rgb: np.ndarray, black_threshold: int = 10, reference_size: int = FOV_REFERENCE_SIZE) -> dict:
    """Measure field-of-view geometry and image statistics inside the retina.
    
    The image is resized so its longest side equals ``reference_size`` before
    the mask and measurements are computed. This makes the sharpness proxy
    less dependent on native image resolution. Color and brightness
    measurements use pixels inside the detected mask; the Laplacian-variance
    sharpness proxy uses an eroded interior mask so the retinal edge does not
    dominate it.
    
    Args:
        rgb: RGB image array with shape ``(height, width, 3)``.
        black_threshold: Threshold passed to ``estimate_fov_mask``.
            Defaults to 10.
        reference_size: Longest-side size in pixels used for the resized
            measurement image. Defaults to ``FOV_REFERENCE_SIZE``.
            
    Returns:
        Dictionary with the fields in ``FOV_COLUMNS``: retinal area and
        bounding-box fractions, normalized center offsets, mean brightness
        and per-channel color means and standard deviations, and
        Laplacian-variance sharpness. If no retinal field is detected, every
        field is ``NaN``.
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


def image_metadata(image_path: ImagePath, black_threshold: int = 10, include_fov: bool = True, fov_reference_size: int = FOV_REFERENCE_SIZE,) -> dict:
    """Extract whole-image metadata and, optionally, retinal-region metadata.
    
    Whole-image color, brightness, and sharpness statistics include every
    pixel and are computed at native resolution. ``black_border_fraction``
    specifically counts pixels for which all three RGB channels are at or
    below ``black_threshold``; it is a dark-pixel proxy, not a retinal
    segmentation. FOV statistics are added only when ``include_fov`` is true.
    
    Args:
        image_path: Path to an image that Pillow can open.
        black_threshold: Integer threshold from 0 to 255 used for the
            black-pixel fraction and FOV mask. Defaults to 10.
        include_fov: If true, include the ``fov_*`` statistics. Set false to
            compute only whole-image measurements. Defaults to true.
        fov_reference_size: Longest-side pixel size used for FOV
            measurements. For an image already resized to a fixed size, use
            that size to avoid upsampling before measuring sharpness.
            Defaults to ``FOV_REFERENCE_SIZE``.
            
    Returns:
        Dictionary containing ``image_id``, ``image_path``, ``width``,
        ``height``, ``aspect_ratio``, ``file_size_bytes``, whole-image
        brightness and RGB means and standard deviations,
        ``blur_laplacian_variance``, and ``black_border_fraction``. If FOV
        measurements are enabled, also includes every field in ``FOV_COLUMNS``.
        
    Raises:
        ValueError: If ``black_threshold`` is outside ``[0, 255]`` or the
            image cannot be opened or decoded.
        OSError: If the image's file size cannot be read from its path.
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
        record.update(fov_metadata(rgb, black_threshold, fov_reference_size))
    return record


def _safe_metadata(image_path, black_threshold, include_fov, fov_reference_size):
    """Run per-image extraction and return an error as data for batch workers.
    
    Args:
        image_path: Image path passed through to ``image_metadata``.
        black_threshold: Pixel threshold passed through to ``image_metadata``.
        include_fov: Whether to calculate field-of-view measurements.
        fov_reference_size: Measurement scale passed to ``image_metadata``.
        
    Returns:
        ``(record, None)`` on success, or ``(None, message)`` if extraction
        raises ``ValueError``. This form lets the batch builder apply its
        requested error policy after serial or parallel processing.
    """
    try:
        return image_metadata(image_path, black_threshold, include_fov, fov_reference_size), None
    except ValueError as error:
        return None, str(error)


def build_image_metadata( image_paths: Iterable[ImagePath], black_threshold: int = 10, include_fov: bool = True, fov_reference_size: int = FOV_REFERENCE_SIZE,
    on_error: str = "raise", n_jobs: int = 1, show_progress: bool = False, ) -> pd.DataFrame:
    """Extract metadata for a collection of image paths.
    
    Paths are materialized and processed in input order. For parallel
    execution, ``n_jobs > 1`` uses worker processes; otherwise processing is
    serial. Set ``show_progress`` to display a progress bar when ``tqdm`` is
    installed. This batch interface makes it convenient to build a table for
    dataset-level analysis while retaining failures for review.
    
    Args:
        image_paths: Iterable of image paths to inspect.
        black_threshold: Threshold passed to ``image_metadata``.
            Defaults to 10.
        include_fov: Whether to include retinal field-of-view measurements.
            Defaults to true.
        fov_reference_size: Longest-side measurement size passed to
            ``image_metadata``. Defaults to ``FOV_REFERENCE_SIZE``.
        on_error: ``"raise"`` to stop and raise ``ValueError`` on the first
            unreadable image, or ``"skip"`` to omit unreadable images and
            record their paths and messages in the result attributes.
            Defaults to ``"raise"``.
        n_jobs: Number of worker processes when greater than 1. Values of 1
            or less use serial processing. Defaults to 1.
        show_progress: Whether to display a progress bar if ``tqdm`` is
            available. Defaults to false.
            
    Returns:
        DataFrame with one row per successfully processed image in input
        order. Its columns are the keys returned by ``image_metadata``; FOV
        columns are present when ``include_fov`` is true. Failed paths are
        available in ``frame.attrs["failed"]`` as dictionaries containing
        ``"image_path"`` and ``"error"`` (empty when there are no failures).
        
    Raises:
        ValueError: If ``on_error`` is not ``"raise"`` or ``"skip"``, or if
            an image cannot be read and ``on_error="raise"``.
    """
    if on_error not in {"raise", "skip"}:
        raise ValueError("on_error must be 'raise' or 'skip'")
    paths = list(image_paths)
    worker = partial(_safe_metadata, black_threshold=black_threshold, include_fov=include_fov,
                    fov_reference_size=fov_reference_size)
    
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
    """Optionally wrap an iterator in a ``tqdm`` progress bar.
    
    Args:
        iterator: Iterable whose items are yielded unchanged.
        total: Expected number of items, used to size the progress display.
        show_progress: Whether to attempt displaying the progress bar.
        
    Returns:
        The original iterator if progress is disabled or ``tqdm`` is
        unavailable; otherwise, a ``tqdm``-wrapped iterator.
    """
    if not show_progress:
        return iterator
    try:
        from tqdm.auto import tqdm
    except ImportError:  # progress bars are optional
        return iterator
    return tqdm(iterator, total=total, desc="Profiling images")
