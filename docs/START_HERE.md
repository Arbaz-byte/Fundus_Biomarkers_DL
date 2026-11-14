# Project Guide

This guide orients new contributors to the project's goal, repository structure, current status, and safe next steps. Dataset provenance and sharing requirements are documented in [`DATASETS.md`](DATASETS.md) and [`DATA_POLICY.md`](DATA_POLICY.md).

## Project overview

The project develops a reproducible computer-vision workflow for diabetic-retinopathy grading from retinal fundus photographs. It also examines whether acquisition-related image properties (such as resolution and field of view) can act as shortcuts, and evaluates preprocessing choices before model training. Vessel measurements are a later planned extension.

This is a research project, not a clinically validated diagnostic system.

## Active dataset and evaluation

The active source is the [official APTOS 2019 Kaggle competition download](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data), stored locally at:

```text
data/raw/APTOS-2019-kaggle-competition/
    train_images/
    train.csv
    test_images/
    test.csv
```

The training CSV contains labels; the competition test CSV contains image IDs only. Therefore, local model evaluation must use a held-out split made from the labeled training data. Keep that local test partition sealed until final evaluation. The unlabeled competition test set is for predictions or an eligible competition submission, not for local metric calculation.

The earlier third-party copy is superseded. The local source comparison found its labeled images match images in the official training pool; its uploader-defined `test` folder is not the competition test set. Do not mix the old copy into new experiments. See [`DATASETS.md`](DATASETS.md) for the comparison and source terms.

APTOS images, transformed examples, per-image data, split manifests, caches, and image-bearing outputs must remain local and git-ignored. Check [`DATA_POLICY.md`](DATA_POLICY.md) before publishing any dataset-derived material.

## Current project status

Status as of **14 November 2026**:

| Area | Status |
|---|---|
| Dataset provenance and handling | Active source and project restrictions are documented. |
| Day 00 source and dataset audit | Complete locally. The source comparison and inventory, class-balance, and resolution checks have been run; detailed outputs remain local. |
| Train/validation/local-test split | Not yet confirmed for the official source. Create and verify a fresh split from labeled official training data only. |
| Image metadata and shortcut analysis | Not yet rerun against the new split; earlier results are historical. |
| Preprocessing | Existing code and exploration notebooks are starting points, not yet validated on the official images. |
| Model training and final evaluation | Not started for the current official-source baseline. |

Before using a local split manifest, verify its source and assignments. In particular, do not assume an existing `configs/splits/aptos_v1.csv` is valid for the official-source reset. The default configuration also needs review: `configs/default.yaml` currently points to the superseded `data/raw/APTOS-2019/` location.

## Recommended workflow

1. Review the source, terms, and handling rules in [`DATASETS.md`](DATASETS.md) and [`DATA_POLICY.md`](DATA_POLICY.md).
2. Review [`notebooks/00_Dataset_audit.ipynb`](../notebooks/00_Dataset_audit.ipynb) and the related audit code. The audit compares the old labeled folders with official training data and does not inspect official test-image files.
3. Correct and verify the active data path, then create a fixed, reproducible train/validation/local-test split from official labeled training data. Keep its manifest local.
4. Rerun the metadata and shortcut analyses using only the verified split.
5. Reassess crop, padding, resize, and optional contrast methods on the official training images. Rebuild local caches only after preprocessing checks pass.
6. Train and tune using train and validation partitions. Evaluate the selected approach once on the sealed local test partition; use the official unlabeled test set only for predictions.

Record measured outcomes in the relevant documentation, and keep image-level artifacts out of version control.

## Repository layout

| Path | Purpose |
|---|---|
| `configs/` | Project settings and local split-manifest location. Verify configuration paths and manifest provenance before running experiments. |
| `data/raw/` | Locally stored source datasets; never commit dataset files. |
| `data/processed/` | Local audit outputs, manifests, per-image tables, QA artifacts, and caches; never commit generated APTOS data. |
| `docs/` | Dataset and data policy, decisions, work log, roadmap, audit checklist, and preprocessing records. `APTOS_PREPROCESSING.md` is active; `PREPROCESSING.md` records work on the superseded source. |
| `notebooks/` | Numbered analyses and experiment narratives. Reusable implementation belongs in `src/` or `scripts/`, not only in notebook cells. |
| `reports/` | Cleared aggregate results and figures only; check data terms before publishing derived results. |
| `scripts/` | Task entry points for source comparison, auditing, split creation, cache generation, and aggregate evaluation. |
| `src/fundus/` | Reusable package code: dataset adapters, duplicate grouping, metadata features, metrics, preprocessing, caching, and split helpers. |
| `tests/` | Automated checks for audit, caching, source comparison, image metadata, preprocessing, shortcut analysis, duplicates, and splits. |

The repository root contains the package and environment configuration (`pyproject.toml`, `requirements.txt`), code licence (`LICENSE`), and the main overview (`README.md`). The code licence applies to code, not to APTOS data.

## Documentation map

- [`README.md`](../README.md): project overview, licence distinction, setup, and high-level status.
- [`ROADMAP.md`](ROADMAP.md): planned phases, deliverables, and current next steps.
- [`LOG.md`](LOG.md): dated project activity and findings.
- [`DECISIONS.md`](DECISIONS.md): recorded design choices and their rationale.
- [`DATASETS.md`](DATASETS.md) and [`DATA_POLICY.md`](DATA_POLICY.md): source provenance, applicable terms, and sharing boundaries.
- [`AUDIT_CHECKLIST.md`](AUDIT_CHECKLIST.md): checks to complete before trusting or sharing results.
- [`APTOS_PREPROCESSING.md`](APTOS_PREPROCESSING.md): active preprocessing plan and findings for the official source.

## Setup and checks

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

The local data directories are ignored by Git. Before staging changes, confirm that no raw images, processed files, split manifests, or image-bearing notebook outputs are included.

## Terms used in this repository

| Term | Meaning |
|---|---|
| DR grade | APTOS severity label from 0 (no DR) through 4 (proliferative DR). |
| QWK | Quadratic weighted kappa, the primary ordered-grade metric used by the competition. |
| Shortcut | A non-disease image cue that may predict labels, such as acquisition or resolution differences. |
| Local test partition | A held-out subset of labeled official training data used for final local evaluation. It is distinct from Kaggle's unlabeled competition test set. |
| Sealed | Not used for training, preprocessing selection, or hyperparameter tuning before final evaluation. |
| FOV | Field of view: the visible retinal area within the image frame. |
