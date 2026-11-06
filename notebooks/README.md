# Notebooks

Naming: `NN_topic.ipynb`, lowercase, one question per notebook. Reusable code goes to `src/fundus/`; notebooks only call it and interpret results.

## Index

| Notebook | Question | Status |
|---|---|---|
| 01_image_variation | How much do the images differ, and does it line up with the grade? | done |
| 02_metadata_shortcut_test | How well does metadata alone predict the grade? | done |
| 03_preprocessing_design | Which fixed preprocessing do we use, and does it remove the source cue? | planned (T-02, T-03) |
| 04_cnn_fundamentals | Do I understand convolution, shapes and training well enough to debug a CNN? | planned (T-04 to T-06) |
| 05_baseline_transfer_learning | How well does a fine-tuned ResNet do on the frozen split? | planned (T-07) |
| 06_per_source_evaluation | How much of the score survives per source and on the source-matched subset? | planned (T-08) |
| 07_enhancement_ablation | Does CLAHE or local-contrast enhancement help at all, under a leak-free protocol? | planned (Sprint 2) |
| 08 and later | Vessel segmentation, biomarkers, cross-dataset tests | see docs/ROADMAP.md |

## Standards for every notebook

1. Top cell: question, inputs, outputs, rules, run order.
2. One import cell, one configuration cell. Paths relative to the project root; print project-relative paths only.
3. Section order: what we do, code, observation, interpretation. Numbers in text must match the printed output.
4. Use the sealed test partition for nothing.
5. Save aggregate results to `reports/` (never images or patient-level rows).
6. Before committing: Restart kernel and Run All; check text against output; make sure no personal paths (search for `/Users/`) and no dataset images remain in outputs unless the licence allows.
