import numpy as np
import pytest
from PIL import Image

from fundus.features.image_metadata import (
    FOV_COLUMNS,
    build_image_metadata,
    estimate_fov_mask,
    image_metadata,
)


def _fake_fundus(path, width, height, radius_fraction=0.45, seed=0):
    """Reddish textured disc on a black background, similar in layout to a fundus photograph."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    centre_y, centre_x = (height - 1) / 2, (width - 1) / 2
    radius = radius_fraction * min(height, width) * 2 / 2 * 1.0
    disc = (yy - centre_y) ** 2 + (xx - centre_x) ** 2 <= radius ** 2
    # smooth texture so that sharpness does not depend on pixel count
    low = rng.normal(size=(24, 24)).astype(np.float32)
    texture = np.asarray(Image.fromarray(low).resize((width, height), Image.BICUBIC))
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.clip(150 + 30 * texture, 0, 255)
    image[..., 1] = np.clip(80 + 20 * texture, 0, 255)
    image[..., 2] = np.clip(30 + 10 * texture, 0, 255)
    image[~disc] = 0
    Image.fromarray(image).save(path)
    return disc.mean()


def test_original_columns_are_unchanged(tmp_path):
    _fake_fundus(tmp_path / "a.png", 600, 400)
    record = image_metadata(tmp_path / "a.png", include_fov=False)
    assert "fov_area_fraction" not in record
    assert record["width"] == 600 and record["height"] == 400
    assert record["aspect_ratio"] == pytest.approx(1.5)


def test_fov_mask_matches_disc_area(tmp_path):
    true_fraction = _fake_fundus(tmp_path / "a.png", 900, 600)
    record = image_metadata(tmp_path / "a.png")
    assert record["fov_area_fraction"] == pytest.approx(true_fraction, abs=0.03)
    assert record["fov_bbox_height_fraction"] > 0.8


def test_fov_stats_are_not_diluted_by_border(tmp_path):
    _fake_fundus(tmp_path / "a.png", 900, 600)
    record = image_metadata(tmp_path / "a.png")
    # inside the disc the image is bright; whole-image mean is pulled down by the black border
    assert record["fov_brightness_mean"] > record["brightness_mean"] * 1.1


def test_fov_sharpness_is_comparable_across_resolution(tmp_path):
    _fake_fundus(tmp_path / "small.png", 600, 400)
    _fake_fundus(tmp_path / "large.png", 2400, 1600)
    small = image_metadata(tmp_path / "small.png")
    large = image_metadata(tmp_path / "large.png")
    native_ratio = small["blur_laplacian_variance"] / large["blur_laplacian_variance"]
    fov_ratio = small["fov_blur_laplacian_variance"] / large["fov_blur_laplacian_variance"]
    # the native-resolution measure depends strongly on pixel count; the FOV measure much less
    assert abs(np.log(fov_ratio)) < abs(np.log(native_ratio))


def test_all_black_image_gives_nan_fov(tmp_path):
    Image.fromarray(np.zeros((100, 120, 3), dtype=np.uint8)).save(tmp_path / "black.png")
    record = image_metadata(tmp_path / "black.png")
    assert all(np.isnan(record[column]) for column in FOV_COLUMNS)


def test_build_skips_unreadable_images(tmp_path):
    _fake_fundus(tmp_path / "ok.png", 300, 200)
    (tmp_path / "bad.png").write_bytes(b"not an image")
    with pytest.raises(ValueError):
        build_image_metadata([tmp_path / "ok.png", tmp_path / "bad.png"])
    frame = build_image_metadata([tmp_path / "ok.png", tmp_path / "bad.png"], on_error="skip")
    assert list(frame["image_id"]) == ["ok"]
    assert len(frame.attrs["failed"]) == 1


def test_parallel_build_matches_serial_and_keeps_order(tmp_path):
    paths = []
    for index, (width, height) in enumerate([(300, 200), (400, 400), (500, 300)]):
        path = tmp_path / f"img{index}.png"
        _fake_fundus(path, width, height, seed=index)
        paths.append(path)
    serial = build_image_metadata(paths, n_jobs=1)
    parallel = build_image_metadata(paths, n_jobs=2)
    assert list(serial["image_id"]) == list(parallel["image_id"])
    assert np.allclose(serial["fov_area_fraction"], parallel["fov_area_fraction"])
