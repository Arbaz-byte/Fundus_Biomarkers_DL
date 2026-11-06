# Preprocessing notes

Reference code: `src/fundus/preprocessing.py` (tested in `tests/test_preprocessing.py`). Baseline pipeline: `crop_to_fov` -> `pad_to_square` -> `resize_square`, wrapped by `preprocess_basic(rgb, size=256)`.

Rule for this document: every claim needs a number or a reproducible demonstration (a notebook cell or a script). Items marked **[demo needed]** still have none in the repository.

## 1. Crop the black border

- **What it is:** estimate the retinal disc with a threshold mask (on a copy resized to 512 px) and crop to its bounding box, rounded outward when mapped back to full resolution.
- **What it solves:** the fixed input resolution is spent on retina (vessels, lesions) instead of black frame. It also keeps colour statistics clean, because the border no longer dilutes them (notebook 01: whole-image blue mean vs border fraction rho -0.42).
- **How much border there is:** in the saved development metadata, the dark-pixel fraction runs from about 0% to 53% (median 20%); about 60% of images have at least 20% dark pixels. It differs a lot by camera: about 1% at 640 x 480, 9% at 2588 x 1958, and about 52% at 2048 x 1536 and 819 x 614.
- **Why it mattered as a shortcut:** the two resolutions with the largest border (2048 x 1536 and 819 x 614) are exactly the two that are 100% and 97% "No DR". Border size was therefore a source signature. Cropping removes it as a *direct* cue. It does not remove the source (see "Residual source cues").
- **Limit:** a full circle in a square still leaves about 21% corner black (1 - pi/4 = 0.215). Only a circular mask removes that.
- **Decision:** keep `crop_to_fov`.

## 2. Pad to square (not stretch)

- **What it is:** add black to the shorter side, centred, so the image becomes square before resizing.
- **What it solves:** stretching distorts disc shape, vessel angles and lesion shape. Example: a crop of 1562 x 1199 would be stretched by a factor 1.30 (1562 / 1199). Padding keeps all geometry.
- **Cost, and a risk:** padding re-creates black bars whose size depends on the camera's crop. For that 1562 x 1199 crop, 23% of the padded height is black ((1562 - 1199) / 1562). A camera with a nearly square crop gets almost none. This is a possible residual source cue, tested in notebook 03 (T-03).
- **Decision:** keep `pad_to_square` with `fill=0`. Never stretch.

## 3. Resize with area interpolation

- **What it is:** `INTER_AREA` when shrinking, `INTER_LINEAR` when enlarging.
- **What it solves:** shrinking by 4 to 8 times with nearest, linear or cubic interpolation skips most source pixels, which causes aliasing (moire, broken or beaded vessels, lesions appearing by chance). Area averaging uses every source pixel.
- **Evidence from the stripe test:** area gave the correct grey (78 vs true 77); cubic overshot (87); the others showed fake bands. **[demo needed]** put the stripe test in notebook 03 so anyone can reproduce these numbers.
- **Decision:** keep as written. Use `INTER_NEAREST` for masks or labels. Use the same interpolation in training and inference.

## 4. Local-average subtraction (Graham-style)

- **What it is:** `out = 4*img - 4*local_average + 128`, with a large Gaussian blur as the lighting estimate.
- **What it solves:** uneven illumination and camera differences. In the test, a lighting slope of about 30 to 95 became flat (about 130), leaving vessels and lesions.
- **Problem found (5 Nov):** a plain blur mixes the black border into the average. Measured on a flat synthetic disc: the inner rim came out at about 245 instead of 128 (saturating to 255), and the border just outside the disc came out at about 47 instead of 0.
- **Fix in the reference module:** mask-normalised blur (blur of `image * mask` divided by blur of `mask`), result re-masked, so the border stays 0 and the rim stays 128. Tested (`test_local_average_subtraction_flat_retina_is_mid_grey_with_no_rim_and_black_outside`). The parameter is named `sigma_fraction`, as in the docstring.
- **Decision:** optional variant, not the default. Apply after resizing to a fixed size.

## 5. CLAHE on the lightness channel

- **What it is:** contrast-limited adaptive histogram equalisation on L in LAB, colour kept.
- **What it solves:** local low contrast and dim regions; the clip limit stops noise in flat areas from being amplified. It only partly flattens uneven lighting.
- **Bug found (5 Nov):** a version of `clahe_l_channel` that ends with `cv2.COLOR_RGB2LAB` instead of `cv2.COLOR_LAB2RGB` gives washed-out pink-grey output. On a reddish test retina, channel means went from (164, 77, 34) to (161, 113, 119) with the wrong call, whereas the correct call gives (175, 86, 42).
- **Status:** the reference module ends with `COLOR_LAB2RGB`, and `test_clahe_keeps_colour_roughly_unchanged_on_a_uniform_image` catches the mistake. If your working copy still ends with `RGB2LAB`, replace it.
- **Decision:** optional variant, not the default.

## Contrast variants: status

Neither is applied by default. Pick one as the main normalisation rather than stacking both, and compare them later under the leak-free protocol (Sprint 2, T-10 to T-13). Do not choose by how the pictures look.

## Residual source cues after preprocessing (to be tested in T-03, section 8)

Notebook 01 showed that the image source is visible in resolution, border size, colour statistics and framing. Preprocessing removes some of this and keeps some:

| Cue | Removed? | Why |
|---|---|---|
| Pixel count (resolution) | yes | everything is resized to one size |
| Black border size | mostly | cropped away, but see padding |
| Padding bars | **no, re-introduced** | their size depends on the camera's crop aspect |
| Colour cast and illumination | no | only the optional variants touch them |
| Noise and sharpness at 256 px | partly | resizing smooths; camera noise patterns may survive |
| Shape of the retinal disc (clipped circle) | no | cropping keeps the disc shape |

T-03 measures how much of the source signal survives by repeating the metadata shortcut test on cached 256 px images, including the dark-pixel fraction per resolution group after crop and pad.

## Open items

- [ ] Reproducible stripe test for section 3 (notebook 03).
- [ ] Decide whether to apply a circular mask after resizing (removes the 21% corner black).
- [ ] Make sure only one module exists for this code (`preprocessing.py`); delete any duplicate `preprocess.py` after diffing.
