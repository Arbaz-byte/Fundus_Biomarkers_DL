"""
Deterministic preprocessing utilities for color fundus photographs.

The baseline pipeline is ``crop_to_fov`` -> ``pad_to_square`` ->``resize_square`` and is available as ``preprocess_basic``. 
Cropping removes the unused dark border, padding preserves the retinal field's aspect ratio, and resizing produces a fixed-size 
input for downstream models.

``local_average_subtraction`` and ``clahe_l_channel`` are optional contrast adjustments; they are not included in the baseline pipeline. These functions
do not use labels. Image-processing functions expect RGB ``uint8`` arrays with shape ``(height, width, 3)``. ``fov_bounding_box`` returns coordinates,
and ``_resize_for_mask`` returns a resized image and scale rather than an RGB image alone.

"""
from typing import Optional, Tuple
import cv2
import numpy as np
from fundus.features.image_metadata import estimate_fov_mask

MASK_REFERENCE_SIZE = 512  # the retina mask is estimated on a copy whose longest side is this many pixels


def _check_rgb(rgb: np.ndarray) -> None:
    """Raise ``ValueError`` unless ``rgb`` is an RGB ``uint8`` image array.
    Args:
        rgb: Array expected to have shape ``(height, width, 3)`` and dtype
            ``numpy.uint8``.
    Returns:
        None.
    Raises:
        ValueError: If the array does not have the expected shape or dtype.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 array of shape (height, width, 3)")



def _resize_for_mask(rgb: np.ndarray, reference_size: int) -> Tuple[np.ndarray, float]:
    """
    Resize an image so its longest side matches a mask-estimation scale.
    Shrinking uses area interpolation and enlarging uses linear interpolation.
    Keeping mask estimation near a consistent resolution makes the fixed-size
    morphology kernels in ``estimate_fov_mask`` behave consistently.

    Args:
        rgb: RGB image array to resize.
        reference_size: Desired length, in pixels, of the resized image's
            longest side.
    Returns:
        A tuple containing the resized RGB image and the scale factor
        ``reference_size / max(original_height, original_width)``.
        
    """
    
    height, width = rgb.shape[:2]
    scale = reference_size / max(height, width)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(rgb, new_size, interpolation=interpolation), scale


def fov_bounding_box( rgb: np.ndarray, black_threshold: int = 10, reference_size: int = MASK_REFERENCE_SIZE ) -> Optional[Tuple[int, int, int, int]]:
    """
    Find the retinal field's bounding box in the original image.
    The field-of-view(fov) mask is estimated after resizing a copy of the image to
    ``reference_size`` on its longest side. The resulting box is mapped back
    to the original resolution and rounded outward to avoid cutting off
    retinal pixels. ``bottom`` and ``right`` are exclusive, so the returned
    coordinates can be used directly as ``rgb[top:bottom, left:right]``.

    Args:
        rgb: RGB ``uint8`` image with shape ``(height, width, 3)``.
        
        black_threshold: Brightness threshold used to distinguish candidate
            field-of-view pixels from the dark border. Defaults to 10.
            
        reference_size: Longest-side size used when estimating the mask.
            Defaults to ``MASK_REFERENCE_SIZE``.
    Returns:
        ``(top, bottom, left, right)`` coordinates in original-image pixels,
        or ``None`` if no retinal field is detected. The box helps crop away
        surrounding black space before resizing model inputs.
    Raises:
        ValueError: If ``rgb`` is not an RGB ``uint8`` image array.
    """
    _check_rgb(rgb)
    height, width = rgb.shape[:2]

    # resize image 
    small, scale = _resize_for_mask(rgb, reference_size)
    mask = estimate_fov_mask(small, black_threshold)
    
    # If retina is not found 
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
    """
    Crop an image to the detected retinal field of view.
    Removing unused dark borders gives downstream models a more consistent
    view of the retina. If no field is detected, the original image is
    returned unchanged.

    Args:
        rgb: RGB ``uint8`` image with shape ``(height, width, 3)``.
        black_threshold: Brightness threshold used to estimate the retinal
            field. Defaults to 10.
    Returns:
        The cropped RGB image, or ``rgb`` unchanged if no field is detected.
    Raises:
        ValueError: If ``rgb`` is not an RGB ``uint8`` image array.
    """
    box = fov_bounding_box(rgb, black_threshold)
    if box is None:
        return rgb
    top, bottom, left, right = box
    
    return rgb[top:bottom, left:right]


def pad_to_square(rgb: np.ndarray, fill: int = 0) -> np.ndarray:
    """Pad an RGB image to a square without changing its aspect ratio.
    Padding rather than stretching preserves retinal shape before resizing.
    If the input is already square, it is returned unchanged. For odd padding
    widths, the extra pixel is placed on the bottom or right.
    
    Args:
        rgb: RGB ``uint8`` image with shape ``(height, width, 3)``.
        fill: Pixel value used for all three color channels in the padding.
            Defaults to 0 (black).
    Returns:
        A square RGB ``uint8`` image containing the original image centered
        within the padding.
    Raises:
        ValueError: If ``rgb`` is not an RGB ``uint8`` image array.
    """
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
    """
    Resize a square RGB image to a requested fixed resolution.
    Area interpolation is used when shrinking and linear interpolation when
    enlarging. Requiring square input helps ensure callers pad rather than
    distort non-square images.
    Args:
        rgb: Square RGB ``uint8`` image with shape ``(side, side, 3)``.
        size: Positive output side length in pixels.
    Returns:
        RGB ``uint8`` image with shape ``(size, size, 3)``.
    Raises:
        ValueError: If ``rgb`` is not a valid RGB ``uint8`` image, is not
            square, or ``size`` is not positive.
    """
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
    
    """
    Apply the baseline crop, square-padding, and resize pipeline.
    This standardizes image dimensions while preserving retinal aspect ratio:
    the field is cropped, the shorter dimension is padded, and the result is
    resized. Contrast enhancement is intentionally not applied.
    
    Args:
        rgb: RGB ``uint8`` image with shape ``(height, width, 3)``.
        size: Positive output width and height in pixels. Defaults to 256.
        black_threshold: Brightness threshold used to detect the retinal
            field. Defaults to 10.
    Returns:
        Preprocessed RGB ``uint8`` image with shape ``(size, size, 3)``.
    Raises:
        ValueError: If ``rgb`` is invalid or ``size`` is not positive.
    """
    
    return resize_square(pad_to_square(crop_to_fov(rgb, black_threshold)), size)


def local_average_subtraction( rgb: np.ndarray, sigma_fraction: float = 1 / 30, mask: Optional[np.ndarray] = None, black_threshold: int = 10, ) -> np.ndarray:
    
    """
    Reduce illumination variation by subtracting a local average. The transform is ``4 * image - 4 * local_average + 128`` inside the
    retinal mask, rounded and clipped to ``[0, 255]``; pixels outside the mask are set to black. 
    
    The local average is a mask-normalized Gaussian blur, which prevents the dark image border from biasing values near the retinal
    edge. If no mask pixels are found, the whole image is treated as the retinal field.
    
    The Gaussian sigma is ``max(1, image_width * sigma_fraction)``. Applying this to fixed-size images makes the effect more comparable across images.
    This optional contrast adjustment is separate from the baseline pipeline.
    
    Args:
        rgb: RGB ``uint8`` image with shape ``(height, width, 3)``.
        sigma_fraction: Gaussian sigma as a fraction of image width.
            Defaults to ``1 / 30``.
        mask: Optional spatial mask identifying retinal pixels. If omitted,
            it is estimated using ``black_threshold``.
        black_threshold: Brightness threshold used when estimating the mask.
            Defaults to 10.
    Returns:
        Contrast-adjusted RGB ``uint8`` image with the same shape as ``rgb``.
        
    Raises:
        ValueError: If ``rgb`` is not an RGB ``uint8`` image array.
        
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
    
    """
    Enhance local contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization) on the image's lightness channel.
    The RGB image is converted to CIELAB, CLAHE is applied only to the lightness channel, and the result is converted back to RGB. Keeping the
    chromatic channels unchanged during equalization helps retain the original color characteristics. This optional adjustment is not part of ``preprocess_basic``.
    
    Args:
        rgb: RGB ``uint8`` image with shape ``(height, width, 3)``.
        clip_limit: CLAHE contrast clipping limit. Defaults to 2.0.
        tile_grid: Number of tiles in the horizontal and vertical directions.
            Defaults to ``(8, 8)``.
    Returns:
        CLAHE-adjusted RGB ``uint8`` image with the same shape as ``rgb``.
    Raises:
        ValueError: If ``rgb`` is not an RGB ``uint8`` image array.
    """
    
    _check_rgb(rgb)
    
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
