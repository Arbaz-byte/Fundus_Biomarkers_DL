# Fundus_Biomarkers_DL

Deep learning and physics-based biomarker analysis of retinal fundus images, built on **public datasets**.

> Research code, not a medical device. Nothing here is validated for clinical use.

## Goals
1. Reproducible baselines for diabetic retinopathy (DR) grading.
2. Vessel segmentation and interpretable biomarkers (tortuosity, calibre, fractal dimension, branching).
3. Cross-dataset generalisation and shortcut/artifact audit.
4. A pipeline that can later run on a clinical dataset without code changes.

## Status
See [docs/LOG.md](docs/LOG.md) for the dated work log.

| Phase | Focus | Status |
|---|---|---|
| 1 | CNN basics + APTOS baseline | not started |
| 2 | Training pipeline | not started |
| 3 | Vessel segmentation (DRIVE) | not started |
| 4 | Biomarkers + statistics | not started |
| 5 | Generalisation + audit | not started |
| 6 | Packaging + write-up | not started |

## Repository layout
```
configs/     YAML configs (paths, image size, seeds, split rules)
data/        local data only, git-ignored (see data/README.md)
docs/        data notes, licences, work log, design decisions
notebooks/   exploration and one notebook per phase
reports/     result summaries and figures (no patient-level data)
scripts/     command-line entry points (train, evaluate, export)
src/fundus/  library code: datasets, models, features, metrics, splits
tests/       unit tests (splits, metrics, adapters)
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Data
Datasets are **not** redistributed here. Download them yourself and read each licence.
See [docs/DATASETS.md](docs/DATASETS.md) and [data/README.md](data/README.md).

## Design principles
- One adapter per dataset, all returning the same fields (`path`, `label`, `patient_id`, `site`).
- Splits are done at patient level, with a duplicate-image check.
- Everything configurable through `configs/`; no hard-coded paths.
- Evaluation can output **aggregate metrics only**, so it can run on sensitive data without exporting it.

## Licence
Code: MIT (see LICENSE). Datasets keep their own licences.
