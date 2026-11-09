"""Index raw fundus images and build a preprocessed image cache.

The cache stores cropped, square-padded, fixed-size PNGs for the selected
manifest splits. By default only training and validation images are selected;
the test split requires explicit opt-in to help prevent accidental access
during model development. Labels are not read or used.
"""
import time
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from fundus.preprocessing import fov_bounding_box, preprocess_basic

DEFAULT_SPLITS = ("train", "val")


def index_raw_images(raw_root, pattern: str = "*/*/*.png") -> Dict[str, Path]:
    
    """Build a lookup from each raw image ID to its file path.
    
    Image IDs are taken from the filename stem. The default glob pattern
    matches layouts such as
    ``<raw_root>/<partition>_images/<partition>_images/<id>.png``.
    
    Args:
        raw_root: Root directory containing raw images.
        pattern: Glob pattern, relative to ``raw_root``, used to find image
            files. Defaults to ``"*/*/*.png"``.
            
    Returns:
        A dictionary mapping each image's file stem to its ``Path``. An empty
        dictionary is returned when no paths match.
        
    Raises:
        ValueError: If multiple matched files have the same file stem,
            because a stem would not uniquely identify a raw image.
    """
    index: Dict[str, Path] = {}
    
    for path in sorted(Path(raw_root).glob(pattern)):
        if path.stem in index:
            raise ValueError(f"Image ID appears in more than one folder: {path.stem}")
        index[path.stem] = path
    return index


def cache_images( manifest: pd.DataFrame, raw_index: Dict[str, Path], out_dir, size: int = 256, splits: Sequence[str] = DEFAULT_SPLITS, black_threshold: int = 10,
    overwrite: bool = False, allow_test: bool = False, limit: int = 0, progress: bool = False, ) -> pd.DataFrame:
    
    """Preprocess selected manifest images and save them as PNG cache files.
    
    Rows are filtered to ``splits`` and sorted by image ID. Each source image
    is decoded as RGB, measured for its native size and field-of-view crop
    box, and passed through ``preprocess_basic`` when a cache file needs to be
    written. 
    
    Existing cache files are left untouched unless ``overwrite`` is
    true. The returned metadata is computed from the current source image
    even when its cache file already exists.
    
    The test partition is excluded by default. Allow it only for final
    evaluation to reduce the risk of accessing test data during development.
    This function reads image IDs and split names from the manifest, not
    labels.
    
    Args:
        manifest: DataFrame containing ``"id_code"`` and ``"split"`` columns.
        raw_index: Mapping from image IDs to their source image paths, usually
            produced by ``index_raw_images``.
        out_dir: Directory in which to save cached ``<id_code>.png`` files.
            It is created if it does not exist.
        size: Positive side length, in pixels, of each preprocessed square
            image. Defaults to 256.
        splits: Split names to process. Defaults to ``("train", "val")``.
        black_threshold: Brightness threshold used to detect the retinal
            field of view. Defaults to 10.
        overwrite: If true, replace existing cache files; otherwise preserve
            them. Defaults to false.
        allow_test: If true, permit processing the ``"test"`` split.
            Defaults to false.
        limit: If positive, process only the first ``limit`` selected IDs
            after sorting; 0 or a negative value processes all selected rows.
            Useful for a small dry run. Defaults to 0.
        progress: If true, display a progress bar when ``tqdm`` is available.
            Defaults to false.
            
    Returns:
        A DataFrame with one row per processed manifest row and these columns:
        ``image_id``, ``split``, ``native_width``, ``native_height``,
        ``mask_found``, ``crop_top``, ``crop_bottom``, ``crop_left``,
        ``crop_right``, ``crop_fraction``, and ``seconds``. If no retinal
        field is found, the crop coordinates are ``-1`` and ``crop_fraction``
        is NaN. ``seconds`` measures decoding, crop-box calculation, optional
        preprocessing, and saving.
        
    Raises:
        KeyError: If the manifest lacks a required column.
        ValueError: If ``"test"`` is selected while ``allow_test`` is false,
            or if ``size`` is invalid for preprocessing.
        FileNotFoundError: If a selected manifest ID has no entry in
            ``raw_index``.
        OSError: If a source image cannot be opened, decoded, or written.
    """
    splits = tuple(splits)

    if "test" in splits and not allow_test:
        raise ValueError("The test partition is sealed; pass allow_test=True only for the final evaluation.")
    rows = manifest[manifest["split"].isin(splits)].sort_values("id_code")

    if limit > 0:
        rows = rows.head(limit)
    missing = sorted(set(rows["id_code"]) - raw_index.keys())

    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest IDs have no PNG, e.g. {missing[:5]}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iterator: Iterable = rows.itertuples(index=False)
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, total=len(rows), desc="Caching")
        except ImportError:
            pass

    records = []
    for row in iterator:

        started = time.perf_counter()
        with Image.open(raw_index[row.id_code]) as image:
            rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]

        box = fov_bounding_box(rgb, black_threshold)
        out_path = out_dir / f"{row.id_code}.png"

        if overwrite or not out_path.exists():
            Image.fromarray(preprocess_basic(rgb, size, black_threshold)).save(out_path)

        if box is None:
            top = bottom = left = right = -1
            crop_fraction = float("nan")
        else:
            top, bottom, left, right = box
            crop_fraction = (bottom - top) * (right - left) / (height * width)

        records.append({
            "image_id": row.id_code, "split": row.split, "native_width": width, "native_height": height,
            "mask_found": box is not None, "crop_top": top, "crop_bottom": bottom, "crop_left": left,
            "crop_right": right, "crop_fraction": crop_fraction, "seconds": time.perf_counter() - started,
        })
    return pd.DataFrame(records)
