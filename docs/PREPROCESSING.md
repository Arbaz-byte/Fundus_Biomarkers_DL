# Preprocessing notes

> **Archived evidence (14 Nov 2026):** all dataset-specific counts, image reviews, resolution examples, cue measurements, and empirical conclusions in this file were produced from the superseded MariaHerreroT third-party copy. They remain here as historical notes only and must not be used as evidence for the official Kaggle download. For active work, use [`APTOS_PREPROCESSING.md`](APTOS_PREPROCESSING.md).

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


## Retina detection review (T-02, finished 7 Nov)

Thirty development images were inspected: one representative per resolution (17), the 6 with the smallest and the 6 with the largest detected retina area (`data/processed/qa/fov_mask_review.csv`, local only).

| Verdict | Images | What it looked like |
|---|---|---|
| OK | 15 | outline follows the retinal circle, crop box contains the retina |
| BORDERLINE | 9 | minor contour irregularities near the frame edge or corners; the crop box is still right. All 9 are representatives of resolutions where the retina nearly fills the frame (detected area 0.72 to 0.99) |
| FAIL | 6 | detected area exactly 1.0, mask covers the whole frame, crop box is 0.999 of the image. All six are 2588 x 1958 |

**What the failures mean.** There is no dark border to separate the retina, so the threshold mask is the whole frame and the crop is a no-op. That is harmless for cropping, but `fov_area_fraction`, `black_border_fraction` and every `fov_*` colour statistic then describe the whole frame, not a retinal disc. The later full development audit in notebook 03 found 168 of 2,963 images (5.7%) with a saturated mask: 132/460 at 2588 x 1958, 35/35 at 640 x 480, and 1/7 at 1476 x 1117. Within-grade rates are 6.4%, 5.2%, 5.4%, 4.2%, and 3.1% for grades 0 through 4, respectively. Within 2588 x 1958, rates range from 47.3% for grade 0 to 5.4% for grade 3; this is descriptive and resolution-dependent, not evidence that grade causes saturation. Two further cautions: (1) `nlargest` breaks ties by row order, so the original six failures were an arbitrary subset of all images with area 1.0; (2) coverage and dark fraction are almost the same measurement (Spearman -0.99), so they are not independent evidence and both lose information when there is no border.

**Decision (D-013).** Keep the images and current crop behavior, and persist `fov_saturated` as a derived per-image column in the existing metadata CSV, computed by the shared metadata builder. Do not add it to the frozen split manifest or feed it to the model. Exclude saturated rows from analyses that interpret `fov_*` measurements as retinal-only. The builder now writes the flag; the regenerated development CSV contains 2,963 rows and 168 saturated flags. Notebook 03 reports rates by grade and resolution and provides diagnosis-blinded overlays for all 168 images. Notebooks 01 and 03b mask or exclude saturated rows from FOV summaries and representative selection; notebook 02 uses a separate non-saturated cohort and duplicate-group folds for FOV model comparisons; notebook 03c uses a paired cohort whose raw and cached masks are both unsaturated.

**Overlay review (12 Nov).** Seven contact sheets show all 168 saturated images with the detected contour and crop box, without diagnosis labels. Many 2588 x 1958 images visibly have a circular field even though the threshold mask covers the frame. The 640 x 480 images are tightly framed, often with the field touching the image edges. These patterns do not establish that one circle-fit fallback will work across both acquisition groups.

**Candidate fallback evaluation (notebook 03d).** A fixed OpenCV Hough-circle configuration was run on all 168 saturated images without loading diagnosis labels. It found candidates for 141 images (122/132 at 2588 x 1958, 18/35 at 640 x 480, and 1/1 at 1476 x 1117); 27 had no candidate. Only 15 candidate circles were fully inside the image frame. Review of all seven diagnosis-blinded candidate sheets showed inconsistent circle size and placement, including circles extending beyond the frame and circles too small or offset to cover the retinal field. **Do not adopt this candidate or change the shared crop behavior.** This rejects the tested Hough configuration, not every possible fallback method. Its review sheets are generated locally under `data/processed/qa/`.

**Housekeeping found in the review sheet.** One verdict was typed with a leading space (" OK") and was not counted; the summary cell now strips spaces.

## Exploration kept, not deleted

CLAHE settings (clip limit, tile grid) and the resize comparison are explored in `notebooks/03b_contrast_exploration.ipynb` (status: exploratory, no decision). It records the parameters tried, shows panels for three images and reports label-free statistics (contrast, clipping, high-frequency energy, a Frangi vessel-response proxy, colour shift). It does not choose a variant: the choice is made in Sprint 2 by training with each variant under a leak-free protocol. A previous experiment cell was removed from notebook 03 and left a cell that depended on deleted code (`results`, `OUT_DIR`, `SIZE`), which made the notebook impossible to run from a clean kernel. Rule from now on: move experiments to an exploratory notebook; never delete them.

## Cache

`python scripts/cache_preprocessed.py --size 256` writes `data/processed/aptos_256px_basic/<id>.png` for the train and validation images only (the test partition is refused unless `--allow-test` is given) and a `_cache_manifest.csv` with native size, crop box, fraction of the frame kept and seconds per image. Notebook 03c verifies the cache and measures how much of the source cue survives.

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

- [ ] Reproducible stripe test for section 3 (notebook 03c or 03b).
- [x] Add `fov_saturated` to the metadata builder and regenerate the existing image-metadata CSV.
- [x] In notebook 03, report saturation rates within grade and resolution, and visually review overlays for all saturated images.
- [x] Evaluate a candidate fallback mask on the full saturated subset; the tested Hough-circle configuration was not reliable and was not adopted.
- [ ] Decide whether to apply a circular mask after resizing (removes the 21% corner black).
- [ ] Make sure only one module exists for this code (`preprocessing.py`); delete any duplicate `preprocess.py` after diffing.
