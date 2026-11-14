# Fundus_Biomarkers_DL

Deep learning and physics-based biomarker analysis of retinal fundus images, using the official APTOS 2019 Kaggle competition download under its competition rules.

> Research code, not a medical device. Nothing here is validated for clinical use.

## Status (re-baselined 2026-11-14)

| Phase | Focus | Status |
|---|---|---|
| 0 | Repository and data policy | done; Day 00 source audit complete |
| 1a | Image-variation audit and metadata shortcut test (notebooks 01, 02) | rerun required on official competition data |
| 1b | Preprocessing and baseline CNN | prior data-dependent results superseded; restart after dataset intake audit |
| 2 | Training pipeline, imbalance handling, enhancement ablation | not started |
| 3 | Vessel segmentation (DRIVE) | not started |
| 4 | Vessel biomarkers and statistics | not started |
| 5 | Cross-dataset generalisation and artifact audit | not started |
| 6 | Packaging and write-up | not started |

The task board with owners, dates and acceptance criteria is in [docs/ROADMAP.md](docs/ROADMAP.md). The dated work log is [docs/LOG.md](docs/LOG.md); design choices are in [docs/DECISIONS.md](docs/DECISIONS.md).

## Current status: fresh dataset baseline

The active source is the [official APTOS 2019 Kaggle competition download](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data), obtained 14 November 2026 and stored locally under `data/raw/APTOS-2019-kaggle-competition/`. It provides train and test partitions, with no supplied validation set. The local Day 00 audit is complete; create a new local validation split before rerunning notebooks 01 and 02.

The local ID, label, and exact-byte comparison confirms the old copy is the same labeled image pool as official `train_images`; it is not additional data, and its folder named `test` is not the Kaggle competition test set. The metadata audit found class imbalance and grade/resolution association; detailed tables remain local-only under ignored `data/processed/`. Split-dependent results still need to be rerun under the official-source split. See [docs/DATASETS.md](docs/DATASETS.md) and [docs/LOG.md](docs/LOG.md).

## Preprocessing

The current preprocessing implementation and synthetic unit tests are reusable starting points, not yet validated on the official competition images. Re-audit image dimensions, framing, retina masks, resize behavior, and preprocessing choices on the new training set before caching or training. Do not inspect or tune on the official test set. See [docs/APTOS_PREPROCESSING.md](docs/APTOS_PREPROCESSING.md). The old-copy analysis is archived in [docs/PREPROCESSING.md](docs/PREPROCESSING.md).

## Repository layout

```
configs/        default.yaml and local split manifests (git-ignored)
data/           local data only, git-ignored (see data/README.md)
docs/           ROADMAP, LOG, DECISIONS, APTOS_PREPROCESSING, DATASETS, DATA_POLICY, AUDIT_CHECKLIST
                PREPROCESSING is the archived analysis of the superseded source
notebooks/      00_Dataset_audit, 01_image_variation, 02_metadata_shortcut_test, 03_preprocessing_design,
                03b_contrast_exploration (exploratory), 03c_cache_and_cue_test
reports/        aggregate results and figures only (see reports/README.md)
scripts/        compare_aptos_sources.py, audit.py, create_aptos_split.py, evaluate_aggregate.py, cache_preprocessed.py
src/fundus/
  __init__.py
  caching.py          builds the preprocessed image cache (train and val only by default)
  datasets/           dataset adapters
  duplicates.py       near-duplicate image grouping
  features/           image_metadata.py (image and field-of-view measurements), shortcut.py (shortcut-test helpers)
  metrics.py          evaluation metrics
  models/             model code (filled in Sprint 1, T-05)
  preprocessing.py    crop, pad, resize and the optional contrast variants
  splits.py           split helpers
tests/          test_caching, test_image_metadata_fov, test_preprocessing, test_shortcut_and_duplicates, test_splits
.gitignore  LICENSE  pyproject.toml  README.md  requirements.txt
```

Present on a working machine but git-ignored: `FDL-env/`, `.pytest_cache/`, `src/fundus.egg-info/`, `data/`.

## Setup

```bash
python -m venv .venv                # Python 3.10 or newer
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

## Reproduce the results so far

1. Read [docs/DATASETS.md](docs/DATASETS.md) and review notebook 00's local-only audit.
2. Create and verify a local train/validation split from official labeled training data only; keep the official test set sealed.
3. Rerun notebooks 01 and 02 against the new split; then re-evaluate preprocessing in notebook 03.
4. Rebuild local caches and rerun downstream analyses only after the intake, split, and preprocessing checks pass.

## Design principles

- One adapter per dataset; every adapter returns `path`, `label`, `patient_id`, `site`.
- Splits are frozen in a manifest file. The **test partition is sealed** and used once, at the end.
- APTOS has no patient IDs, so its splits are image-level. Near-duplicate images are kept together.
- Metrics: quadratic weighted kappa (primary), macro-F1, balanced accuracy. Measure class balance on the active source before selecting baselines or interpreting accuracy.
- Evaluation scripts can output aggregate metrics only, so they can run on sensitive data without exporting it.

## Limitations

- Image-level splits only; images from the same patient may sit in different partitions.
- The official Kaggle competition rules govern the active download. Do not publish APTOS images or transformed examples; keep per-image data and split manifests local. Aggregate results, figures, model artifacts, and code sharing have separate conditions; see [docs/DATASETS.md](docs/DATASETS.md).
- Metadata analyses are indirect probes of shortcut risk, not proof about any CNN.

## Licence

**Code:** MIT (see [LICENSE](LICENSE)); this covers code only. For code that qualifies as Kaggle Competition Code, rule 8 also describes where public code must be shared and requires an OSI-approved licence permitting commercial use; MIT meets the licence-type condition but does not by itself meet the Kaggle sharing-location condition.

**Data:** The active dataset is the official APTOS 2019 Kaggle competition download. Its use is governed by the [competition rules](https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules), including non-commercial-use and data-security restrictions. Do not publicly display source or transformed images without explicit written permission. MIT applies to code only, not dataset contents.
