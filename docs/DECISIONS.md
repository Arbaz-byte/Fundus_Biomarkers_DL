# Design decisions

One entry per decision. These are drafts based on the work so far: **rewrite the "Why" in your own words** and only mark an entry Accepted when you can defend it without looking at the notebook.

| ID | Date | Decision | Why | Evidence | Status |
|---|---|---|---|---|---|
| D-001 | 2026-10-20 | Public repository; no clinical data ever committed. | Portfolio value, nothing sensitive to protect, and no way to leak by accident. | docs/DATA_POLICY.md | Accepted |
| D-002 | [date] | Primary metric is quadratic weighted kappa (QWK); also report macro-F1 and balanced accuracy. | Grades are ordered; 51.5% of images are "No DR", so accuracy alone is misleading. | Notebook 01, class balance | Accepted |
| D-003 | [date] | One frozen stratified image-level split (`aptos_v1`, seed 42) stored in `configs/splits/`. The provider's own partitions are ignored. The test partition stays sealed. | One fixed benchmark prevents re-splitting after seeing results. APTOS has no patient IDs, so patient-level splitting is impossible. | `scripts/create_aptos_split.py` [confirm proportions and near-duplicate handling here] | Accepted |
| D-004 | 2026-10-28 | Near-duplicates are grouped (perceptual hash distance at most 4 and SSIM at least 0.95 on 128 x 128 versions) and kept in the same CV fold. | Copies of one photograph must not sit on both sides of a split. | Notebook 02, section 2 | Accepted |
| D-005 | 2026-10-28 | Run a metadata-only shortcut test before training any CNN. | If metadata alone predicts the grade, headline CNN scores need a different protocol. | Notebook 02: CV QWK 0.698, validation 0.692 | Accepted |
| D-006 | 2026-10-30 | For image-content questions prefer `fov_*` measurements (inside the retinal disc, fixed 512 px) over whole-image ones. | Whole-image colour is diluted by the black border; native Laplacian variance tracks pixel count (rho -0.74 with width). | Notebook 01, section 6 | Accepted; revisit after seeing the printed FOV correlations |
| D-007 | 2026-10-30 | Do not report a headline CNN score from the random split alone. Always add per-resolution results and a source-matched subset (2588 x 1958 and 2416 x 1736). | Resolution is a strong label proxy in this file. | Notebook 01, section 5; notebook 02, section 6 | Proposed |
| D-008 | 2026-10-30 | Do not download the 18.5 GB 2015 + 2019 dataset before Phase 5. | Slows Phase 1; contents and licence unverified; needed only for the cross-dataset study. | docs/DATASETS.md | Proposed |
| D-009 | [date] | Baseline preprocessing is the simplest option (crop to the retinal disc, pad to a square, resize, ImageNet normalisation); enhancement variants are compared later under a leak-free protocol. | Choosing a variant by looking at pictures or by one validation score invites self-deception. | Notebook 03 (to be written) | Proposed |

| D-011 | 2026-11-05 | Baseline pipeline is crop to the retinal disc, pad to a square with black, resize with area interpolation. Never stretch. | [your own words: geometry is kept, resolution is spent on retina, aliasing avoided] | docs/PREPROCESSING.md | Accepted |

| D-012 | 2026-11-05 | Local-average subtraction and CLAHE stay optional variants, not defaults. The first uses a mask-normalised blur. Pick one main variant later under the leak-free protocol, not by looking at pictures. | [your own words] | docs/PREPROCESSING.md | Accepted |