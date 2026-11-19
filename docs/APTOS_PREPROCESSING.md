# APTOS 2019 preprocessing — official Kaggle download

This is the active preprocessing record for `data/raw/APTOS-2019-kaggle-competition/`. Dataset terms and sharing restrictions are summarized in [`DATASETS.md`](DATASETS.md).

## Status

The official download has `train_images/` with `train.csv`, and `test_images/` with `test.csv`; it has no supplied validation split. The train CSV has `id_code` and `diagnosis`; the test CSV has `id_code` only. Local source and image-metadata audits report no inventory, label, byte-hash, or measured-metadata mismatches between the old labeled folders and official training images. The old uploader's provenance and licence remain unverified. Per-image and resolution-by-grade tables remain local-only.

The approved local evaluation plan is to split only the labeled official training data into approximately 70% train, 15% validation, and 15% local test. Use a fixed seed and diagnosis-stratified grouping so confirmed exact and near-duplicate images stay together. The validation partition is for model and preprocessing choices; reserve the local test for one final evaluation after those choices are fixed. The official unlabeled competition test set remains sealed.

These are approximate proportions: duplicate groups cannot be divided, so realized sizes and grade proportions must be checked and recorded. APTOS provides no patient IDs, so duplicate grouping does not establish patient-level independence. The grade imbalance means per-grade estimates, especially for smaller grades, may be noisy; a split cannot guarantee zero bias. Report resolution distributions by partition as a shortcut diagnostic, but do not force resolution balancing without justification. Keep the per-image split manifest under git-ignored `data/processed/`.

All evidence in [`PREPROCESSING.md`](PREPROCESSING.md) belongs to the superseded MariaHerreroT copy. Do not copy its image counts, QA outcomes, mask rates, or chosen parameters into this document as current findings.

## Safe evaluation sequence

1. Run `python scripts/compare_aptos_sources.py` to compare old-copy versus official-training IDs, labels, and exact file-byte SHA-256 hashes. It checks the official `test.csv` schema and IDs but never reads or hashes official test images. Its per-ID report defaults to git-ignored `data/processed/aptos_source_comparison.csv`.
2. Run `python scripts/audit.py` or `notebooks/00_Dataset_audit.ipynb` to audit CSV schemas, image inventories, class balance, native resolutions, and resolution-by-grade counts. The audit profiles only the old labeled folders and official training images; its per-image tables are git-ignored. Notebook 00 loads existing audit outputs by default and supports an explicit `FORCE_REBUILD` to rerun profiling; it presents aggregate tables and plots only.
3. Review the class and resolution outputs locally. Treat any association between image acquisition metadata and grade as a potential shortcut, not a clinical conclusion. Do not publish the aggregate findings until sharing is cleared under the competition rules.
4. Create and verify approximate 70/15/15 train/validation/local-test partitions from the labeled official training data. Stratify by diagnosis while keeping confirmed duplicate groups intact. Record actual row, group, and per-grade counts; verify no duplicate group crosses partitions. Keep the official test set sealed.
5. Review representative training images locally across the observed resolution and framing groups. Save image-bearing QA outputs only under git-ignored `data/processed/`.
6. Evaluate the existing crop, pad, and resize functions against the new training images and synthetic tests. Measure mask failures, geometry preservation, and output quality before choosing or changing a pipeline.
7. Rebuild the cache and repeat the source-cue analysis using only the verified train and validation partitions. Do not use the local test partition to tune preprocessing, select models, or set thresholds.
8. Record measured results and decisions here. Do not publish APTOS images or transformed examples; see [`DATASETS.md`](DATASETS.md).

## Candidate implementation

The current code in `src/fundus/preprocessing.py` provides `crop_to_fov`, `pad_to_square`, `resize_square`, and `preprocess_basic`. These functions are a starting point, not a validated pipeline for the official package. Preserve aspect ratio unless experiments show a justified alternative. Treat the current 256-pixel size, interpolation, mask threshold, and optional enhancement methods as candidates until re-evaluated.

## Findings

The source-comparison report establishes that the old labeled image files are byte-identical to images in official `train_images`, with matching labels. The old copy's `test` partition is not the official competition test set. The local metadata audit found grade imbalance and a resolution-grade association; treat the latter as a potential shortcut, not a clinical conclusion. Exact class and resolution counts remain in the git-ignored audit outputs and are not cleared for public release.

## Open tasks

- [x] Rewrite `scripts/create_aptos_split.py` to read only official `train.csv` and `train_images/`, emit a local git-ignored 70/15/15 manifest, and report actual split/class/group counts.
- [x] Add focused tests for official-source indexing, deterministic assignments, class stratification, duplicate-group isolation, and validation of malformed/missing inputs before generating the production manifest.
- [ ] Run the focused split tests in the configured project environment, inspect actual split/class/group counts, and review the generated local manifest before using it.
- [ ] Review the local resolution-by-grade table and decide which source-cue checks should be run after the official-source split is fixed.
- [ ] Review retina-mask and crop QA on a representative, training-only sample.
- [ ] Validate preprocessing determinism, output shape, value range, and geometry on the new source.
- [ ] Rebuild the cache and rerun the preprocessing cue test after the split is fixed.
