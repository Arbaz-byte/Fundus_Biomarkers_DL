# data/

This folder is git-ignored. Never commit images, and never place clinical or patient data in this repository (see `docs/DATA_POLICY.md`).

## Layout used now

```
data/raw/APTOS-2019/               downloaded dataset (pre-split download)
    train_images/train_images/*.png
    val_images/val_images/*.png
    test_images/test_images/*.png
    train_1.csv  valid.csv  test.csv       original provider files (not used for our split)
data/processed/
    aptos_development_image_metadata.csv   written by notebook 01 (train + validation images only)
    aptos_development_duplicate_groups.csv written by notebook 02 (near-duplicate groups, cache)
    aptos_<size>px_<variant>/              preprocessed image cache (written by scripts/cache_preprocessed.py)
```

The partition of each image (train / val / test) comes only from `configs/splits/aptos_v1.csv`, not from the provider's folders. The sealed test partition is not opened by any notebook.

## Rules

- Everything in `data/processed/` can be regenerated; delete it freely.
- Do not copy dataset images into `docs/`, `reports/` or notebook outputs unless the dataset licence allows it.
- Record every new dataset in `docs/DATASETS.md` (source, licence, download date) before using it.
