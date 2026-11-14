# data/

Raw and processed data are local-only and git-ignored. Never commit dataset images or per-image outputs (see [`docs/DATA_POLICY.md`](../docs/DATA_POLICY.md)).

## Active raw dataset

```text
data/raw/APTOS-2019-kaggle-competition/
    train_images/
    train.csv
    test_images/
    test.csv
```

This is the official APTOS 2019 Kaggle competition download, obtained 14 November 2026. The download has train and test partitions, with no supplied validation split. The training CSV is labeled; the test CSV contains IDs only. The local intake audit found matching CSV/image inventories and an imbalanced grade distribution with resolution differences across grades. Detailed counts and per-image metadata stay in ignored `data/processed/aptos_dataset_audit/`. See [`docs/DATASETS.md`](../docs/DATASETS.md) for provenance and terms, and [`docs/APTOS_PREPROCESSING.md`](../docs/APTOS_PREPROCESSING.md) for the active preprocessing audit.

The official test partition stays sealed. Create the project's train/validation split from labeled training data only, and save its manifest locally; never mix the old third-party copy into the new analysis.

## Previous source (superseded)

`data/raw/APTOS-2019/` contains the older MariaHerreroT third-party copy with pre-split train, validation, and test folders. Its upload terms were not verified in this project. The local source comparison confirms its labeled image bytes and labels match the official competition training set; its `test` folder is not the Kaggle competition test set. This does not establish uploader provenance or licence. Keep the roots separate and use only the official root for new experiments.

## Processed data

`data/processed/` is for regenerable local-only artifacts, including the comparison report, audit metadata tables, duplicate groups, split manifests, preprocessed caches, QA sheets, and per-image predictions. Keep all APTOS-derived rows and uncleared aggregate outputs local.

## Rules

- Both raw-data folders and processed outputs must remain ignored by Git.
- `.gitkeep` files only preserve empty directory structure; no dataset files are allowed through ignore rules.
- Do not copy APTOS images or image-bearing outputs into `docs/`, `reports/`, or notebooks.
- Record every new dataset's source, terms, download date, contents, and limitations in [`docs/DATASETS.md`](../docs/DATASETS.md) before analysis.
