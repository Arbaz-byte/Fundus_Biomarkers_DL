# Fundus_Biomarkers_DL

Deep learning and physics-based biomarker analysis of retinal fundus images, built on **public datasets**.

> Research code, not a medical device. Nothing here is validated for clinical use.

## Status (updated 2026-10-30)

| Phase | Focus | Status |
|---|---|---|
| 0 | Repository, data policy, frozen split manifest (`aptos_v1`) | done |
| 1a | Image-variation audit and metadata shortcut test (notebooks 01, 02) | done, re-run pending |
| 1b | Preprocessing pipeline and baseline CNN (notebooks 03 to 06) | not started |
| 2 | Training pipeline, imbalance handling, enhancement ablation | not started |
| 3 | Vessel segmentation (DRIVE) | not started |
| 4 | Vessel biomarkers and statistics | not started |
| 5 | Cross-dataset generalisation and artifact audit | not started |
| 6 | Packaging and write-up | not started |

The task board with owners, dates and acceptance criteria is in [docs/ROADMAP.md](docs/ROADMAP.md). The dated work log is [docs/LOG.md](docs/LOG.md); design choices are in [docs/DECISIONS.md](docs/DECISIONS.md).

## Key finding so far: the APTOS-2019 file contains a source shortcut

In the APTOS-2019 download used here, **image resolution is strongly linked to the DR grade**. For example, all 291 images at 2048 x 1536 are "No DR", while only 2 of 311 images at 3216 x 2136 are. The 2,963 development images come in 17 distinct resolutions, and 6 of them (46% of the images) are at least 90% a single grade.

| Metadata-only probe (no retinal content) | QWK |
|---|---|
| Grouped 5-fold CV on train, gradient boosting, 13 whole-image measurements | 0.698 (SD 0.012) |
| Same model, validation partition (525 images) | 0.692 |
| Lookup table: most common training grade at the image's resolution, validation | 0.594 |
| Guessing (dummy baselines) | about 0 |

<!-- Update these numbers after re-running notebook 02 if your output differs. -->

QWK is quadratic weighted kappa (0 = chance, 1 = perfect agreement on an ordered scale). The probe uses **no image content beyond file-level measurements**, so a model trained on a random split of this pool may partly learn the image source instead of the disease. This is a property of the dataset; whether a CNN exploits it is tested in notebooks 05 and 06. See [notebook 01](notebooks/01_image_variation.ipynb) and [notebook 02](notebooks/02_metadata_shortcut_test.ipynb).

## Repository layout

```
configs/        YAML configs and frozen split manifests (configs/splits/aptos_v1.csv)
data/           local data only, git-ignored (see data/README.md)
docs/           roadmap, work log, decisions, dataset notes, data policy
notebooks/      numbered analysis notebooks (see notebooks/README.md)
reports/        aggregate results and figures only (see reports/README.md)
scripts/        command-line entry points (split creation, aggregate evaluation)
src/fundus/     library code
  datasets/       dataset adapters
  features/       image metadata (image_metadata.py) and shortcut-test helpers (shortcut.py)
  duplicates.py   near-duplicate image grouping
  splits.py       patient-level splitting helpers
  metrics.py      evaluation metrics
tests/          unit tests
```

## Setup

```bash
python -m venv .venv                # Python 3.10 or newer
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

## Reproduce the Phase 1a results

1. Download APTOS-2019 into `data/raw/APTOS-2019/` (see [docs/DATASETS.md](docs/DATASETS.md); datasets are not redistributed here).
2. `python scripts/create_aptos_split.py` creates `configs/splits/aptos_v1.csv` (already committed; re-running with the same seed should give the same file).
3. Run `notebooks/01_image_variation.ipynb`, then `notebooks/02_metadata_shortcut_test.ipynb`, top to bottom.

## Design principles

- One adapter per dataset; every adapter returns `path`, `label`, `patient_id`, `site`.
- Splits are frozen in a manifest file. The **test partition is sealed** and used once, at the end.
- APTOS has no patient IDs, so its splits are image-level. Near-duplicate images are kept together.
- Metrics: quadratic weighted kappa (primary), macro-F1, balanced accuracy. Plain accuracy is not used alone, because 51.5% of images are "No DR".
- Evaluation scripts can output aggregate metrics only, so they can run on sensitive data without exporting it.

## Limitations

- Image-level splits only; images from the same patient may sit in different partitions.
- The origin of the pre-split APTOS download and its exact licence terms are still to be confirmed (see docs/DATASETS.md).
- Metadata analyses are indirect probes of shortcut risk, not proof about any CNN.

## Licence

Code: MIT (see LICENSE). Datasets keep their own licences and are not part of this repository.
