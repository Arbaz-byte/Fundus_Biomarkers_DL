import numpy as np
import pytest

from fundus.preprocessing import (
    clahe_l_channel,
    crop_to_fov,
    fov_bounding_box,
    local_average_subtraction,
    pad_to_square,
    preprocess_basic,
    resize_square,
)


def _disc_image(width, height, centre_x, centre_y, radius, seed=0):
    """Reddish textured disc on a black background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    disc = (yy - centre_y) ** 2 + (xx - centre_x) ** 2 <= radius ** 2
    image = np.zeros((height, width, 3), dtype=np.uint8)
    noise = rng.integers(0, 30, size=(height, width), dtype=np.uint8)
    image[..., 0] = 140 + noise
    image[..., 1] = 70 + noise // 2
    image[..., 2] = 30 + noise // 3
    image[~disc] = 0
    return image


def _nonblack_box(image, threshold=10):
    mask = image.max(axis=2) > threshold
    rows, columns = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    return rows[0], rows[-1] + 1, columns[0], columns[-1] + 1


def test_crop_removes_the_black_border():
    image = _disc_image(900, 600, 450, 300, 250)
    cropped = crop_to_fov(image)
    assert cropped.shape[0] == pytest.approx(500, abs=8)
    assert cropped.shape[1] == pytest.approx(500, abs=8)


def test_bounding_box_slices_back_into_the_image():
    image = _disc_image(900, 600, 300, 300, 200)
    top, bottom, left, right = fov_bounding_box(image)
    assert 0 <= top < bottom <= 600 and 0 <= left < right <= 900
    assert left == pytest.approx(100, abs=6) and right == pytest.approx(500, abs=6)


def test_crop_of_a_fully_bright_image_is_unchanged():
    image = np.full((256, 256, 3), 120, dtype=np.uint8)
    assert crop_to_fov(image).shape == image.shape


def test_all_black_image_is_returned_unchanged():
    black = np.zeros((100, 120, 3), dtype=np.uint8)
    assert fov_bounding_box(black) is None
    assert crop_to_fov(black).shape == black.shape


def test_pad_to_square_is_symmetric_black_and_keeps_content():
    image = np.full((100, 160, 3), 200, dtype=np.uint8)
    padded = pad_to_square(image)
    assert padded.shape == (160, 160, 3)
    assert padded[:30].sum() == 0 and padded[-30:].sum() == 0          # black bars top and bottom
    assert (padded[30:130] == 200).all()                               # content untouched


def test_resize_square_shape_dtype_and_error_on_non_square():
    image = np.full((300, 300, 3), 50, dtype=np.uint8)
    out = resize_square(image, 128)
    assert out.shape == (128, 128, 3) and out.dtype == np.uint8
    with pytest.raises(ValueError):
        resize_square(np.zeros((10, 20, 3), dtype=np.uint8), 8)


def test_preprocess_basic_centres_an_off_centre_disc():
    image = _disc_image(1200, 800, 300, 400, 280)      # retina far to the left of the frame
    out = preprocess_basic(image, size=256)
    assert out.shape == (256, 256, 3)
    top, bottom, left, right = _nonblack_box(out)
    assert (left + right) / 2 == pytest.approx(128, abs=6)
    assert (top + bottom) / 2 == pytest.approx(128, abs=6)


def test_preprocess_basic_does_not_stretch():
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    image[150:250, 150:450] = 180                       # a 300 x 100 bright rectangle: aspect ratio 3
    out = preprocess_basic(image, size=150)
    top, bottom, left, right = _nonblack_box(out)
    assert (right - left) / (bottom - top) == pytest.approx(3.0, rel=0.15)


def test_preprocess_is_deterministic():
    image = _disc_image(700, 500, 350, 250, 220, seed=3)
    assert np.array_equal(preprocess_basic(image, 200), preprocess_basic(image, 200))


def _flat_disc(size=256, radius=110, colour=(150, 80, 40)):
    yy, xx = np.mgrid[0:size, 0:size]
    disc = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= radius ** 2
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[disc] = colour
    return image, disc


def test_local_average_subtraction_flat_retina_is_mid_grey_with_no_rim_and_black_outside():
    image, disc = _flat_disc()
    out = local_average_subtraction(image)
    yy, xx = np.mgrid[0:256, 0:256]
    distance = np.sqrt((yy - 128) ** 2 + (xx - 128) ** 2)
    assert out.dtype == np.uint8
    # The estimated mask trims 1-pixel tips of the circle, so stay 3 px away from the exact boundary.
    assert np.abs(out[distance <= 107].astype(int) - 128).max() <= 1   # no bright rim or dark ring at the edge
    assert out[distance >= 113].max() <= 1                              # border stays black, not grey


def test_local_average_subtraction_flattens_an_illumination_gradient():
    image, disc = _flat_disc()
    gradient = np.linspace(0.4, 1.0, image.shape[1], dtype=np.float32)[None, :, None]
    shaded = (image * gradient).astype(np.uint8)
    out = local_average_subtraction(shaded)
    assert out[disc][:, 0].std() < shaded[disc][:, 0].std() * 0.35


def test_clahe_keeps_shape_and_dtype_and_is_deterministic():
    image = _disc_image(256, 256, 128, 128, 120, seed=5)
    out = clahe_l_channel(image)
    assert out.shape == image.shape and out.dtype == np.uint8
    assert np.array_equal(out, clahe_l_channel(image))


def test_functions_reject_non_rgb_input():
    with pytest.raises(ValueError):
        pad_to_square(np.zeros((10, 10), dtype=np.uint8))


def test_clahe_keeps_colour_roughly_unchanged_on_a_uniform_image():
    # Regression test: ending with the wrong colour conversion gives washed-out purple/pink output.
    uniform = np.full((64, 64, 3), (180, 90, 40), dtype=np.uint8)
    out = clahe_l_channel(uniform)
    assert np.abs(out.astype(int) - uniform.astype(int)).max() <= 12
    assert out[..., 0].mean() > out[..., 1].mean() > out[..., 2].mean()    # still red > green > blue
