# Notebooks

Naming: `NN_topic.ipynb`, lowercase, one question per notebook; the requested Day 00 entry is `00_Dataset_audit.ipynb`. Reusable code goes to `src/fundus/`; notebooks only call it and interpret results.

## Index

| Notebook | Question | Status |
|---|---|---|
| [00_Dataset_audit](00_Dataset_audit.ipynb) | Do the old and official sources reconcile, and how do grades and native resolutions relate? | audit complete locally; outputs ignored; review before 01 |
| 01_image_variation | How much do the images differ, and does it line up with the grade? | rerun required after 00 and a new official-source split |
| 02_metadata_shortcut_test | How well does metadata alone predict the grade? | rerun required; previous outputs superseded |
| 03_preprocessing_design | Does retina detection work across the new training images? | rerun required; prior QA superseded |
| 03b_contrast_exploration | How do CLAHE settings and other variants change label-free statistics? | exploratory; rerun on new source before using |
| 03c_cache_and_cue_test | Is the cache correct, what does it cost, and how much of the source cue survives preprocessing? | rerun required after new split and cache |
| 04_cnn_fundamentals | Do I understand convolution, shapes and training well enough to debug a CNN? | planned (T-04 to T-06) |
| 05_baseline_transfer_learning | How well does a fine-tuned ResNet do on the frozen split? | planned (T-07) |
| 06_per_source_evaluation | How much of the score survives per source and on the source-matched subset? | planned (T-08) |
| 07_enhancement_ablation | Does CLAHE or local-contrast enhancement help at all, under a leak-free protocol? | planned (Sprint 2) |
| 08 and later | Vessel segmentation, biomarkers, cross-dataset tests | see docs/ROADMAP.md |

## Standards for every notebook

1. Top cell: question, inputs, outputs, rules, run order.
2. One import cell, one configuration cell. Paths relative to the project root; print project-relative paths only.
3. Section order: what we do, code, observation, interpretation. Numbers in text must match the printed output.
4. Use official test images for nothing until final evaluation; derive validation only from labeled training data.
5. Save only publication-cleared aggregate results to `reports/`. Keep APTOS-derived per-image tables and any uncleared aggregate outputs in git-ignored `data/processed/`.
6. Before committing: Restart kernel and Run All; check text against output; make sure no personal paths (search for `/Users/`) and clear all dataset-image outputs. The Kaggle competition rules do not provide a small-sample exception for image examples. Use synthetic or separately licensed images for public demonstrations.

## Exploration policy

Trying settings and plotting them is real experimentation and is encouraged. Rules: keep the code (never delete an experiment; move it to a notebook named `NNb_<topic>`), mark the notebook "exploratory, no decision", write down the parameters tried, save only aggregate results, and make the *choice* later with a leak-free protocol (see docs/ROADMAP.md, Sprint 2). A notebook must always run from a clean kernel: no cell may depend on a variable defined in deleted code.

