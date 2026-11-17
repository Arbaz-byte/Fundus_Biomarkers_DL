# Work log

One entry per working day: what I did, what broke, what I learned, what is next. Fill the bracketed parts yourself; they are about your days, which I cannot see.

> Entries through 13 Nov describe the MariaHerreroT third-party pre-split copy unless stated otherwise. Their data-dependent findings are historical and superseded, not results for the active official Kaggle competition download.

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

## 2026-11-05 to 2026-11-06
- T-02: reviewed preprocessing.py function by function; documented in docs/PREPROCESSING.md; fixed local-average subtraction (mask-normalised blur) and checked CLAHE colour conversion; FOV mask review: 15 OK, 9 borderline, 6 fail (all 2588x1958, mask = whole frame).

## 2026-11-07
- Ticket: T-03
- Done: YES
- Saturated masks (area >= 0.99): [count, by resolution, share by grade]
- Cache: [images, minutes, MB]; cue test: [grade QWK raw vs cached; source balanced accuracy vs chance]
- Next: T-04

## 2026-11-08 to 2026-11-09
- On vacation, no work.

Next action on return (Tue 10 Nov): re-entry check, then [T-03 unfinished items / T-04].


## 2026-11-12
- Block 0: orientation. Read START_HERE and the guided tour.
- Theory covered so far: CNN basics (filters, feature maps, pooling) and autograd only.
- Not covered yet: loss and softmax in depth, backprop derivation, Dataset/DataLoader, optimization.
- Questions I still have: [write them here]
- Next: Sat 14 Nov, Block 1 (backprop and autograd from scratch).

## 2026-11-14 — Day 00: official-source reset
- Downloaded the official APTOS 2019 Kaggle competition package from the [competition data page](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) into `data/raw/APTOS-2019-kaggle-competition/`.
- The official `train.csv` contains labels; `test.csv` contains IDs only. There is no supplied validation folder.
- The local source-comparison report confirms that IDs, labels, and exact file bytes in the old labeled folders match official `train_images`. The old copy's `test` folder is part of official training data, not the competition test set. This does not verify the old uploader's provenance or licence.


## 2026-11-17 — Day 00: audit workflow and documentation complete

- Completed `scripts/compare_aptos_sources.py` and `scripts/audit.py` for reconciling the old copy with the official APTOS download and profiling labeled images. The official test CSV schema/IDs may be inspected, but official test-image files are never opened, hashed, decoded, or profiled.
- Added documentation to both scripts and the focused tests in `tests/test_compare_aptos_sources.py`, `tests/test_audit.py`, and `tests/test_image_metadata_fov.py`, clarifying purpose, inputs, outputs, test setup, and expected behavior.
- Completed `notebooks/00_Dataset_audit.ipynb` as the aggregate-only audit walkthrough. It loads existing audit outputs when available, supports an explicit `FORCE_REBUILD` refresh, and presents aggregate tables and plots without saving image examples or per-image output in the notebook.
- The local audit found no CSV/image inventory, ID, label, byte-hash, or measured-metadata mismatches. It confirms class imbalance and a resolution-grade association that I will treat as a potential shortcut, not a clinical conclusion. Detailed tables remain local under git-ignored `data/processed/aptos_dataset_audit/`.
- Python syntax checks and `git diff --check` passed for the documentation updates. Pytest was not rerun as part of this docstring pass. Before committing, clear local notebook outputs and keep audit tables and per-image data out of Git.
