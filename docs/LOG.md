# Work log

One entry per working day: what I did, what broke, what I learned, what is next. Fill the bracketed parts yourself; they are about your days, which I cannot see.

## 2026-10-20
- Created the repository (public) with the structure, MIT licence, data policy and `.gitignore`.
- Added topics and pinned the repository.

## 2026-10-25 to 2026-10-27
- Downloaded APTOS-2019 (9 GB, pre-split download) into `data/raw/APTOS-2019/`.
- [Describe: how the split was created, how long the download took, any problems.]
- Added `scripts/create_aptos_split.py` and `configs/splits/aptos_v1.csv`.

## 2026-10-28
- First versions of the image-variation and metadata-shortcut notebooks executed.
- Main result: metadata alone predicts the grade (CV QWK 0.698, validation 0.692); resolution alone gives 0.594 as a lookup table.
- [What I learned / what surprised me.]

## 2026-10-30
- Restructured into two notebooks; moved reusable code to `src/fundus/` (`features/image_metadata.py`, `features/shortcut.py`, `duplicates.py`) with tests.
- Added field-of-view measurements (`fov_*`) because whole-image colour and native sharpness are tied to border size and pixel count.
- Cleaned stale outputs and re-ran both notebooks. [Record: how long profiling took, whether any numbers in the text differed from the output and what I changed.]
- [Record what the FOV correlations showed: do they remove the tie to width and border?]
- Next: T-02, preprocessing design (docs/ROADMAP.md).

## 2026-11-05
- Ticket: T-02 (carried over from 6 Nov)
- Done: reviewed every function in preprocessing.py; wrote docs/PREPROCESSING.md; found and fixed two problems (plain-blur border artifact in local-average subtraction, wrong colour conversion in CLAHE); added tests.
- FOV mask review: [OK / BORDERLINE / FAIL counts, failing IDs, or "not done yet"]
- Learned: padding re-creates a camera-dependent black bar, so cropping does not remove the source cue by itself.
- Corrected two sentences in notebook 01 (square images still have about 21% dark corners).
- Next: T-03 (6 Nov)