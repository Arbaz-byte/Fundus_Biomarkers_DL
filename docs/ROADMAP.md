# Roadmap and task board

Working rules, tickets with acceptance criteria, and where every file goes. Owner of every ticket: Arbaz. Budget: 2 hours per night. Dates assume work every night; if a ticket slips, move the next ones and write the reason in `docs/LOG.md`.

## How a ticket is worked

1. **Read** the ticket and the "learn first" list. Spend at most 30 minutes of the night on learning.
2. **Build** in the order given. Code that will be reused goes to `src/fundus/` with a test; notebooks call it and interpret results.
3. **Check** against "Definition of done". If a criterion fails, the ticket is not done.
4. **Record**: `docs/LOG.md` (every day), `docs/DECISIONS.md` (only if you chose something), README status table (when a phase changes). Write decisions in your own words.
5. **Commit** with short imperative messages, for example `Add FOV mask QA overlay`, `Add cached 256px preprocessing script`, `Fix leakage in train loader`.

**Stop rules.** A diagnostic that already gives its answer is finished: do not add machinery. If a ticket takes more than 1.5 times its estimate, stop, write down why, and ask for help before continuing. Run the simple check first (a table or a plot), then build the sophisticated version only if the simple one is not enough.

**Hardware (MacBook, Apple Silicon, 16 GB).** Use PyTorch with the `mps` device (`torch.backends.mps.is_available()`); set `PYTORCH_ENABLE_MPS_FALLBACK=1` for operations MPS does not support. Start at 256 px images and batch size 32; use `num_workers=0` or `2` (macOS starts workers by spawning processes); keep float32 first. Measure and log images per second; that number decides how many experiments you can afford.

---

## Sprint 1: preprocessing and baseline (30 Oct to 8 Nov)

### T-01 Housekeeping and re-run (Fri 30 Oct, about 2 h)

**Goal.** A clean repository whose notebooks reproduce from scratch.

Steps
1. Delete stale files (listed in the chat message), then run `pytest -q`.
2. Add to `requirements.txt`: `imagehash`, `scikit-image`, `seaborn`, `tqdm`. In `pyproject.toml` set `requires-python = ">=3.10"`.
3. Run notebook 01 with Restart kernel and Run All. First run profiles about 3,000 large images and takes several minutes.
4. Run notebook 02 with Restart kernel and Run All.
5. Compare the quoted numbers with your output (checklist below) and edit any sentence that differs.
6. Write the missing interpretation sentences about the `fov_*` results (checklist items 6 and 7).
7. Remove saved outputs that show dataset images (sample-image cell in notebook 01) unless the licence allows them; search the notebooks for `/Users/`.
8. Replace the markdown files from the update package (compare your modified `DATASETS.md` and `LOG.md` first with `git diff`) and fill every bracketed field.
9. Commit in four commits: library code and tests; notebooks; docs; reports.

Checklist of things left for you
1. Metadata CSV header contains `fov_area_fraction` (otherwise notebook 01 did not rebuild it).
2. Notebook 01: 17 resolutions; 6 are at least 90% one grade covering 1,369 images; PCA: resolution explains about 0.79 of PC1 and 0.54 of PC2, grade 0.30 and 0.12.
3. Notebook 02: best CV QWK close to 0.70 and validation close to 0.69; lookup baseline about 0.59.
4. Notebook 02, section 2: note how many duplicate groups span train and validation. If more than zero, read the clean-validation line.
5. `scripts/create_aptos_split.py`: does it group near-duplicates across **all** images, including test, before assigning partitions? If not, note it in DECISIONS (D-003) and fix it before any test evaluation.
6. Notebook 01, section 6: what do the FOV correlations with width and border show? Write 2 to 3 sentences.
7. Notebook 02, section 5: is the `FOV colour + sharpness only` QWK much lower than the whole-image content group, or similar? Write which of the two cases happened and what it means.
8. `tests/test_image_metadata.py` is no longer visible in your tree; if you removed it, fine (the new tests cover the module). There is also no test for `AptosDataset`; add one when T-05 uses it.
9. Ask your clinical collaborator: which clinics/cameras use which image sizes; whether the 2588 x 1958 images with the lowest blue-channel values look different (show him 6 images from each end of the range).
10. Fill `docs/DATASETS.md`: source link of the pre-split upload, licence text, download date.
11. Do **not** download the 18.5 GB 2015 + 2019 dataset yet.

Definition of done: `pytest` green; both notebooks run top to bottom without errors; numbers in text match output; no personal paths; docs updated; four commits pushed.

### T-02 Preprocessing functions and FOV mask QA (Sat 31 Oct, about 2 h)

**Learn first (30 min).** How fundus preprocessing works: cropping the black border, padding to a square so the circle is not stretched, interpolation and anti-aliasing when shrinking large images (INTER_AREA), and the idea behind local-average subtraction (a Gaussian-blurred copy is subtracted to remove illumination differences; associated with the winning solution of the 2015 Kaggle DR competition: search it). Read one short explanation of CLAHE.

**Build.**
1. `src/fundus/preprocessing.py` with `crop_to_fov(rgb, black_threshold=10)`, `pad_to_square(rgb)`, `resize_square(rgb, size)`, and `preprocess_basic(rgb, size)` (crop, pad, resize). Reuse `estimate_fov_mask` from `features/image_metadata.py`. Contrast variants (`local_average_subtraction`, `clahe_l_channel`) as separate functions, not applied by default.
2. `tests/test_preprocessing.py`: output shape is `(size, size, 3)`; a synthetic off-centre disc ends up centred; padding is black; the function does not change a square image that is already cropped.
3. QA: in notebook 03, section 2, overlay the FOV mask outline on one image per resolution (17 images) plus the 6 images with the smallest and 6 with the largest `fov_area_fraction`. Look at every one.

**Expect.** Masks follow the retinal circle on most images. Possible failures: very dark images (mask too small), images where the retina touches the frame edge. If masks fail, adjust the threshold or the morphology and note it in DECISIONS.

**Definition of done:** functions tested; QA grid reviewed; list of failing images (if any) written in LOG.

### T-03 Notebook 03: preprocessing design and cache (Sun 1 Nov, about 2 h)

Notebook `03_preprocessing_design.ipynb`, headings:
1. Purpose, inputs, outputs, rules
2. Setup
3. FOV mask QA (from T-02)
4. Basic preprocessing on one image per resolution (before and after)
5. Contrast variants side by side for one image per grade (raw, local-average subtraction, CLAHE). Visual comparison only; no variant is chosen by looks.
6. Cost: seconds per image and disk size of the cache at 256, 384 and 512 px
7. **Does preprocessing remove the source cue?** Compute fixed-size `fov_*` colour statistics on the cached 256 px images and repeat the metadata shortcut CV from notebook 02. Geometry features no longer exist after resizing; the question is how much of the signal survives in colour and sharpness.
8. Decision: baseline preprocessing = `preprocess_basic` at 256 px, ImageNet normalisation (see D-009)

Also create `scripts/cache_preprocessed.py` (reads `configs/splits/aptos_v1.csv`, writes `data/processed/aptos_256px_basic/<id>.png` for train and val only; the test partition is **not** cached yet).

**Expect.** The cache for about 3,000 images at 256 px is small (hundreds of MB at most) and builds in minutes. The metadata QWK after preprocessing is lower than before but probably not zero; write down the number whatever it is.

Definition of done: cache exists and is git-ignored; section 7 has a number; D-009 filled.

### T-04 Notebook 04, part A: CNN fundamentals by hand (Mon 2 Nov, about 2 h)

**Learn first (45 min).** Stanford CS231n notes on convolutional networks; or the CNN chapters of *Dive into Deep Learning* (d2l.ai). Concepts: convolution as a sliding dot product, output size `floor((n + 2p - k) / s) + 1`, number of parameters of a conv layer, receptive field, pooling, why a global average pool replaces a huge fully connected layer, cross-entropy and what the loss should be at initialisation.

Notebook `04_cnn_fundamentals.ipynb`, headings:
1. Purpose and what you should be able to explain afterwards
2. Output-size and parameter-count formulas, checked against PyTorch on 5 layers you choose
3. Convolution by hand on a 5 x 5 example, then a NumPy `conv2d`, compared with `torch.nn.functional.conv2d`
4. Receptive field of a stack of layers (compute by hand, then in code)
5. A `SmallCNN` (4 conv blocks, global average pool, linear layer): print every tensor shape and the parameter count

**Expect.** NumPy and PyTorch outputs agree to numerical precision. If not, the bug is in your indexing.

Definition of done: you can explain, without looking, why a 3 x 3 conv with 32 filters on a 3-channel input has 896 parameters.

### T-05 Dataset class, model code, overfit test (Tue 3 Nov, about 2 h)

Build
1. `src/fundus/datasets/torch_dataset.py`: `FundusTorchDataset(manifest_csv, cache_dir, split, transform)` returning `(image_tensor, grade)`. Reads only the requested split; asserts that the split name is `train` or `val` unless `allow_test=True`.
2. `src/fundus/models/cnn.py`: `SmallCNN` and `build_resnet(name="resnet18", num_classes=5, pretrained=True)`.
3. `tests/test_torch_dataset.py`: shapes, dtypes, no test images when `split="train"`.
4. Notebook 04, part B: **overfit-one-batch test**: train on 32 images until the loss is close to zero.

**Expect.** Initial loss close to `ln(5) = 1.61`. The model reaches about 100% training accuracy on the 32 images within a few hundred steps. If it cannot, there is a bug in the labels, the loss, the learning rate, or the data loader; fix this before anything else.

### T-06 Training loop and a first small run (Wed 4 Nov, about 2 h)

Build
1. `src/fundus/training.py`: `train_one_epoch`, `evaluate`, device selection (`mps` > `cuda` > `cpu`), seed setting, saving predictions (image ID, true grade, predicted grade, class probabilities) to CSV.
2. Notebook 04, part C: train `SmallCNN` for a few epochs on a 512-image subset at 128 px; plot loss and QWK per epoch; record images per second on your machine.

**Expect.** Loss decreases, validation QWK is clearly above 0, and well below the metadata baseline (0.69). A small CNN from scratch on a small subset is not supposed to be good. The goal is a trustworthy pipeline.

### T-07 Notebook 05: transfer-learning baseline (Thu 5 and Fri 6 Nov, about 4 h)

**Learn first (30 min).** PyTorch tutorial "Transfer Learning for Computer Vision"; why ImageNet weights help; freezing versus fine-tuning; class-weighted loss versus weighted sampling.

Notebook `05_baseline_transfer_learning.ipynb`, headings:
1. Purpose, inputs, outputs, rules (test sealed; validation used for epoch selection only)
2. Setup and configuration (`configs/baseline_resnet18.yaml`: size 256, batch 32, learning rates, epochs, seed)
3. Data loaders: cached 256 px images, flips and rotations only, ImageNet normalisation
4. Model: ResNet18 with ImageNet weights, new 5-class head
5. Training: 2 epochs with the backbone frozen, then fine-tune everything with a lower learning rate; class-weighted cross-entropy; keep the epoch with the best validation QWK
6. Evaluation: QWK, macro-F1, balanced accuracy, per-grade recall, confusion matrix
7. Comparison with the metadata-only references: validation QWK 0.692 (13 features) and 0.594 (resolution lookup)
8. Save `reports/aptos_baseline_resnet18_summary.json` and `reports/preds/aptos_resnet18_val_predictions.csv`
9. Observations and decisions

**Expect.** The CNN should beat the metadata baseline. If it does not, suspect a pipeline bug. If the random-split QWK is very high (around 0.9 or more), do not celebrate: go to T-08 and check how much comes from the source. Run 3 seeds if time allows and report mean and spread. Log images per second and total training time.

### T-08 Notebook 06: per-source evaluation (Sat 7 Nov, about 2 h)

Notebook `06_per_source_evaluation.ipynb`, headings:
1. Purpose: how much of the CNN score survives when the source is controlled
2. Load validation predictions from T-07
3. Metrics per resolution group (accuracy and confusion matrix; QWK is undefined or degenerate inside a source that has only one grade, so say so)
4. The source-matched subset (2588 x 1958 and 2416 x 1736): validation QWK of the model from T-07 on this subset versus on all validation images
5. Train a second model on the source-matched training images only; compare on the same matched validation images
6. Compare both with the metadata-only model inside the matched sources (notebook 02, section 6)
7. Conclusions: the gap, and what it says about the shortcut

**Expect.** The matched-subset score is lower than the all-data score; the size of the gap is the finding. With only 960 images in the matched sources, treat the second model as exploratory and report uncertainty (bootstrap confidence interval over validation images).

### T-09 Sprint review (Sun 8 Nov, about 2 h)

1. Update the README status table and the results table (random split, per source, matched subset, metadata baseline).
2. Write `docs/SPRINT1_SUMMARY.md` (one page for your clinical collaborator: what was done, what was found, what you need from him).
3. Complete LOG and DECISIONS (D-007, D-009 to Accepted or Rejected, with reasons).
4. Tag the repository `v0.1-baseline` and push.

---

## Sprint 2: training pipeline and the enhancement question (9 to 22 Nov)

| Ticket | What | Deliverable |
|---|---|---|
| T-10 | Write the analysis plan **before** running anything: question (does CLAHE or local-contrast enhancement help grading?), variants (raw, local-average subtraction, CLAHE), metric (QWK), seeds, splits, comparison rule | `docs/ANALYSIS_PLAN_enhancement.md` |
| T-11 | Config-driven training script with experiment logging and saved predictions | `scripts/train.py`, `configs/*.yaml` |
| T-12 | Imbalance and augmentation study (class weights versus sampler; flips, rotations, brightness jitter), applied only after splitting | notebook `07_training_choices.ipynb` |
| T-13 | Enhancement ablation: 3 variants x 3 seeds, grouped folds, confidence intervals, per-source results | notebook `08_enhancement_ablation.ipynb` |
| T-14 | Sprint review, README and docs update | `docs/SPRINT2_SUMMARY.md` |

Afterwards, per the master plan: vessel segmentation on DRIVE (Phase 3), biomarkers and statistics (Phase 4), cross-dataset study with the 2015 data (Phase 5), packaging (Phase 6).

---

## Where each file goes

| File | Ticket | Purpose |
|---|---|---|
| `src/fundus/preprocessing.py`, `tests/test_preprocessing.py` | T-02 | crop, pad, resize, contrast variants |
| `scripts/cache_preprocessed.py` | T-03 | writes `data/processed/aptos_256px_basic/` |
| `notebooks/03_preprocessing_design.ipynb` | T-02, T-03 | design and checks |
| `notebooks/04_cnn_fundamentals.ipynb` | T-04 to T-06 | learning notebook |
| `src/fundus/datasets/torch_dataset.py`, `tests/test_torch_dataset.py` | T-05 | PyTorch dataset over the manifest |
| `src/fundus/models/cnn.py` | T-05 | SmallCNN and ResNet builder |
| `src/fundus/training.py` | T-06 | loops, device, predictions |
| `configs/baseline_resnet18.yaml` | T-07 | all baseline settings |
| `notebooks/05_baseline_transfer_learning.ipynb` | T-07 | baseline |
| `reports/aptos_baseline_resnet18_summary.json`, `reports/preds/*.csv` | T-07 | aggregate results, IDs and predictions only |
| `notebooks/06_per_source_evaluation.ipynb` | T-08 | per-source analysis |
| `docs/SPRINT1_SUMMARY.md` | T-09 | one-page summary for collaborator |
| `docs/LOG.md` | every night | work log |
| `docs/DECISIONS.md` | when a choice is made | decision record |

## Risks

| Risk | Mitigation |
|---|---|
| Training is slow on MPS | Start at 128 or 256 px; subsets for debugging; measure images per second first |
| Baseline is below the metadata reference | Treat as a pipeline bug; run the overfit-one-batch test again |
| Baseline looks suspiciously good | Check per-source results and duplicates before believing it |
| Schedule slips because the code is generated faster than understood | Write the DECISIONS entry yourself each time; if you cannot explain a function, rewrite it by hand |
| Test partition used by accident | Dataset class refuses `test` unless `allow_test=True`; used once at the end of a sprint |
