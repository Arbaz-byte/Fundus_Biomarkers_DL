# Current project decisions

This is the short register of decisions that guide work on the active dataset. Superseded analyses and their history remain documented in [`DATASETS.md`](DATASETS.md) and [`LOG.md`](LOG.md), not as active decisions here.

| ID | Date | Decision | Status |
|---|---|---|---|
| D-001 | 2026-11-14 | Use the official APTOS 2019 Kaggle competition download as the sole active dataset. Do not mix it with the older MariaHerreroT upload. | Active |
| D-002 | 2026-11-14 | The official `test.csv` has image IDs but no labels, so it cannot be used for local metric evaluation. Create train, validation, and sealed local-test partitions from the labeled official training data, grouping near-duplicates before splitting. Use validation for model and preprocessing choices; use the local test only for final evaluation. Keep official test images for predictions or an eligible competition submission, not as labeled local test data. | Active |
| D-003 | 2026-11-14 | Treat the Kaggle competition rules as the data-use terms; MIT applies only to project code. Do not publish APTOS images, transformed examples, or per-image data. There is no 10-image exception. | Active |
| D-004 | 2026-11-14 | The local ID/label/SHA-256 report confirms the old copy is the same labeled image pool as official `train_images`, not an independent dataset; its `test` folder is not the competition test set. Keep roots separate, preserve prior results as historical, and reproduce split-dependent results from the official root. This does not verify the old uploader's provenance or licence. | Active |
| D-005 | 2026-11-14 | The labeled grades are imbalanced. Start with an unresampled baseline, report per-grade metrics, and evaluate class weighting only as a validation-controlled experiment. Do not choose imbalance-handling methods using the sealed local-test partition. | Active |
| D-006 | 2026-11-14 | Before analyzing any additional dataset, record its source, applicable terms, download date, and intended use. Do not treat third-party re-uploads of competition data as independent datasets. | Active |
