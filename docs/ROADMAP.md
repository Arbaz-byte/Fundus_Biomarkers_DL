# Roadmap and task board (re-baselined 2026-11-02, updated 2026-11-07)

Owner of every ticket: Arbaz. Budget: 2 hours per night. This board replaces the earlier version after a break (31 Oct to 2 Nov). Everything moves back by about one week; the last phase is shortened by one week instead of working longer hours.

If the 2 hours per night are shared with other projects or study blocks, say so and the board will be re-cut. Do not try to catch up by working extra hours.

## Calendar after the re-baseline

| Phase | Old dates | New dates |
|---|---|---|
| 1 Preprocessing and baseline (Sprint 1) | 25 Oct to 7 Nov | **3 Nov to 14 Nov** |
| 2 Training pipeline, enhancement ablation (Sprint 2) | 8 to 21 Nov | 15 to 28 Nov |
| 3 Vessel segmentation, DRIVE (Sprint 3) | 22 Nov to 5 Dec | 29 Nov to 12 Dec |
| 4 Biomarkers and statistics (Sprint 4) | 6 to 26 Dec | 13 Dec to 2 Jan |
| 5 Cross-dataset study and artifact audit (Sprint 5) | 27 Dec to 10 Jan | 3 to 17 Jan |
| 6 Packaging and write-up (Sprint 6) | 11 to 31 Jan | 18 to 31 Jan (two weeks instead of three) |

Scope cuts agreed to protect the end date (do them only if you fall behind, and log it): three seeds move to Sprint 2; the contrast-variant visual comparison moves to Sprint 2; the matched-subset second model in T-08 stays exploratory.

## How a ticket is worked

1. **Re-entry (10 min, every night):** `git status`, read the last `docs/LOG.md` entry, open the ticket.
2. **Learn** (at most 30 min), **Build**, **Audit**, **Record**. The audit step is part of the ticket, not an extra.
3. **Record:** `docs/LOG.md` every night. `docs/DECISIONS.md` only when you chose something. README status table when a phase changes. Write decisions in your own words.
4. **Commit** with short imperative messages (`Add FOV mask QA overlay`).
5. **Stop rules.** A diagnostic that has given its answer is finished. If a ticket takes more than 1.5 times its estimate, stop, log why, and ask for help. Run the simple table or plot first; build the sophisticated version only if the simple one is not enough. Reusable code goes to `src/fundus/` with a test; notebooks only call it and interpret.
6. **Overflow rule.** If a ticket is not done at the end of its night, it continues the next night and every later ticket moves by one day. Both buffer days of Sprint 1 are already used, so a further slip is absorbed by the scope cuts listed above, not by longer nights.
7. **Exploration is allowed and kept.** Trying settings and plotting them is real experimentation. Put it in an exploratory notebook (`NNb_<topic>`), mark it "exploratory, no decision", write down the parameters tried, time-box it (30 minutes), save only aggregate results, and make the choice later under a leak-free protocol. Never delete experiment code; a notebook must run from a clean kernel.

**Hardware (MacBook, Apple Silicon, 16 GB).** PyTorch with the `mps` device; set `PYTORCH_ENABLE_MPS_FALLBACK=1`. Start at 256 px and batch size 32; `num_workers` 0 or 2; float32 first. Measure and log images per second.

---

## Sprint 1: preprocessing and baseline (3 to 14 Nov)

### T-01 Re-entry, housekeeping, re-run (Tue 3 Nov, 2 h; may continue on Wed)

**Goal.** A clean repository whose first two notebooks reproduce from scratch and whose text matches their output.

**Re-entry check (15 min).** For each item write done / not done in `docs/LOG.md`: stale files deleted; `requirements.txt` and `pyproject.toml` updated; notebook 01 re-run with `fov_*` columns present; notebook 02 re-run; markdown files replaced; sample-image output cleared; commits made.

**Tasks** (skip those already done)
1. Delete stale files: `.pytest_cache/`, `data/processed/aptos_metadata_features_development.csv`, `data/processed/aptos_development_image_metadata.csv`, all generated files in `reports/` and `reports/figures/` except `README.md` and `.gitkeep`.
2. `requirements.txt`: add `imagehash`, `scikit-image`, `seaborn`, `tqdm`. `pyproject.toml`: `requires-python = ">=3.10"`. Run `pytest -q`.
3. Notebook 01: Restart kernel and Run All (several minutes on first run). Notebook 02: same.
4. Write the missing interpretation sentences: notebook 01 section 6 (what the `fov_*` correlations with width and border show) and notebook 02 section 5 (is `FOV colour + sharpness only` much lower than whole-image content, or similar?).
5. Compare every number quoted in the text with your output; correct the text.
6. Fill `docs/DATASETS.md` (source link of your pre-split upload, licence, download date) and rewrite the DECISIONS "Why" cells in your own words.
7. Four commits: code and tests; notebooks; docs; reports.

**Notebook work.** No new notebook. Edit notebooks 01 and 02 only.

**Audit**
- Does `scripts/create_aptos_split.py` group near-duplicates across all images, including test, before assigning partitions? Write the answer in DECISIONS (D-003).
- Notebooks contain no `/Users/` path and no dataset images in saved outputs (unless the licence allows them).
- The sealed test partition is not read anywhere (search the notebooks for `test`).
- Duplicate groups spanning train and validation: record the number.

**Expected result.** Numbers close to: 17 resolutions; 6 of them at least 90% one grade covering 1,369 images; metadata CV QWK about 0.70, validation about 0.69, resolution lookup about 0.59.

**Update docs.** LOG (entry for 2 Nov: write only "no work, break"; 3 Nov: ticket results), DECISIONS (D-003 audit answer; D-006 status after the FOV result), DATASETS.

**Done when.** `pytest` green; both notebooks run top to bottom; text matches output; four commits pushed.

### T-02 Preprocessing functions and FOV mask QA (Thu 5 Nov, carried over from Wed 4 Nov, 2 h)

**Goal.** Tested functions that turn any APTOS image into a fixed-size square, and proof that retina detection works on every resolution family.

**Learn first.** Fundus preprocessing basics: cropping the black border, padding (not stretching) to a square, interpolation and anti-aliasing when shrinking (area interpolation), local-average subtraction (subtract a Gaussian-blurred copy; associated with a winning 2015 Kaggle DR solution, search for it), CLAHE.

**Tasks**
1. Create `src/fundus/preprocessing.py`: `crop_to_fov`, `pad_to_square`, `resize_square`, `preprocess_basic(rgb, size)`; separate `local_average_subtraction` and `clahe_l_channel` that are **not** applied by default. Reuse `estimate_fov_mask`.
2. Create `tests/test_preprocessing.py`: output shape `(size, size, 3)`; an off-centre synthetic disc ends up centred; padding is black; an already-square cropped image stays unchanged.
3. Start `notebooks/03_preprocessing_design.ipynb` (sections 1 to 3 below) and produce the mask overlay grid.

**Notebook 03, first sections**
1. Purpose, inputs, outputs, rules
2. Setup
3. FOV mask QA: outline overlay for one image per resolution (17), plus the 6 images with the smallest and the 6 with the largest `fov_area_fraction`

**Audit**
- Look at every one of the 29 overlay images. List failures (mask too small, retina touching the frame, very dark images) with image ID.
- Preprocessing is deterministic: same input gives identical output.
- Aspect ratio is preserved (no stretching): check on the widest and the squarest image.
- Output values are uint8 in [0, 255].
- `local_average_subtraction` uses the retina mask (flat retina stays 128 up to the edge, border stays 0) and `clahe_l_channel` ends with `COLOR_LAB2RGB` (both tested).
- Padding re-creates black bars whose size depends on the camera crop (for example 23% of the height for a 1562 x 1199 crop). Record this per resolution group; it is a possible residual source cue (see `docs/PREPROCESSING.md`).
- Every claim in `docs/PREPROCESSING.md` has a number or a reproducible demonstration.
- Only one preprocessing module exists in `src/fundus/`.

**Expected result.** Masks follow the retinal circle on most images. A few dark or off-frame images may fail; decide per case whether to adjust the threshold or accept.

**Update docs.** LOG (failure list); DECISIONS (any threshold change); README status table row 1b: in progress.

**Done when.** Functions and tests pass; QA grid reviewed; failure list written.

### T-03 Cache, cost and the source-cue test (Sat 7 Nov, 2 h)

**Goal.** One cached, identically preprocessed copy of all train and validation images, and a measured answer to "how much of the source cue survives preprocessing?"

**Part 0, loose ends of T-02 (30 min).**
1. `git status`: the screenshot of 7 Nov still showed `preprocessing.py` modified, `.gitignore` modified and notebook 03 untracked. Commit what is ready.
2. Make notebook 03 reproducible: Restart kernel and Run All. A cell that depends on deleted code must be removed or moved to an exploratory notebook (see rule 7).
3. Run the new section "How many images have a saturated mask?" and write observation, interpretation and decision in your own words.
4. Record the decision in DECISIONS (D-013).

**Part 1, the cache (25 min).**
1. `python scripts/cache_preprocessed.py --size 256 --limit 20` (dry run), then without `--limit`. Train and validation images only.
2. Read `_cache_manifest.csv`: seconds per image, images with no retina found, crop fractions.

**Part 2, notebook `03c_cache_and_cue_test.ipynb` (50 min).** Sections: check the cache; cost at 256, 384 and 512 px; before/after images; padding bars and dark pixels per resolution; the source-cue test (grade from image statistics, and "which source?" from the same statistics, raw versus cached). Write the observation cells yourself.

**Part 3, record (15 min).** LOG, DECISIONS (D-009, D-013), README status.

**Optional, only if time remains (30 min, time-boxed).** Run `03b_contrast_exploration.ipynb` and write the observation prompts. If it does not fit tonight, it goes to Sprint 2 (T-12); do not extend the night.

**Audit**
- Cache holds exactly the 2,963 development IDs; no test ID; 50 sampled files are 256 x 256 x 3 (checked in 03c).
- The "source" test uses no label; the grade test uses the same locked model and folds as notebook 02.
- Time per image and cache size are logged.

**Expected result.** Cache builds in minutes. Grade QWK from image statistics after preprocessing is lower than from the raw statistics but probably not zero; the source (resolution family) is likely still recognisable well above chance, because colour, noise and padding bars remain. Whatever the numbers are, write them down.

**Update docs.** LOG; DECISIONS D-009 and D-013; README status row 1b.

**Done when.** Cache exists and is git-ignored; `aptos_preprocessing_cue_test.json` saved; observation cells written; decisions recorded.

### T-04 Notebook 04, part A: CNN fundamentals by hand (Sun 8 Nov, 2 h)

**Goal.** Understand shapes, parameter counts and receptive fields well enough to debug a CNN.

**Learn first (45 min).** Stanford CS231n notes on convolutional networks, or the CNN chapters of *Dive into Deep Learning* (d2l.ai). Convolution as a sliding dot product; output size `floor((n + 2p - k) / s) + 1`; parameters of a conv layer; receptive field; pooling; global average pooling; expected loss at initialisation for 5 classes (`ln 5`).

**Notebook `04_cnn_fundamentals.ipynb`, part A**
1. Purpose: what you should be able to explain afterwards
2. Setup
3. Output-size and parameter-count formulas, checked against PyTorch for 5 layers you choose
4. Convolution by hand on a 5 x 5 example; then a NumPy `conv2d` compared with `torch.nn.functional.conv2d`
5. Receptive field of a stack of layers (by hand, then in code)
6. `SmallCNN`: 4 conv blocks, global average pool, linear layer; print every tensor shape and the parameter count

**Audit**
- NumPy and PyTorch outputs agree to numerical precision (`np.allclose`).
- Hand-computed parameter counts equal PyTorch's counts.

**Expected result.** Everything agrees. If not, the mistake is in your indexing or your formula.

**Update docs.** LOG ("what I understood / what is still unclear" in your words).

**Done when.** You can explain, without looking, why a 3 x 3 convolution with 32 filters on a 3-channel input has 896 parameters.

### T-05 Dataset class, model code and the overfit test (Mon 9 Nov, 2 h)

**Goal.** A PyTorch data pipeline that cannot touch the test partition, and a model that can learn.

**Tasks**
1. `src/fundus/datasets/torch_dataset.py`: `FundusTorchDataset(manifest_csv, cache_dir, split, transform)` returns `(image_tensor, grade)`. It asserts `split` is `train` or `val` unless `allow_test=True`.
2. `src/fundus/models/cnn.py`: `SmallCNN` and `build_resnet(name="resnet18", num_classes=5, pretrained=True)`.
3. `tests/test_torch_dataset.py`: shapes, dtypes, labels in 0..4, no test IDs when `split="train"`, train and val IDs disjoint.
4. Notebook 04, part B: overfit-one-batch test.

**Notebook 04, part B**
7. Data loader sanity: show 8 images with labels; print batch shapes and dtypes
8. Initial loss check (should be near 1.61)
9. Overfit 32 images until the loss is close to zero; plot the loss

**Audit**
- Dataset length equals the manifest count for each split.
- Transforms with randomness (flips, rotations) apply to train only; validation uses the same preprocessing without augmentation.
- Overfit test passes. If it does not, fix labels, loss, learning rate or loader before anything else.

**Expected result.** Initial loss close to 1.61; about 100% training accuracy on the 32 images within a few hundred steps.

**Update docs.** LOG; DECISIONS (augmentations chosen for the baseline: flips and rotations only, and why).

**Done when.** Tests pass; overfit test passes.

### T-06 Training loop and a first small run (Tue 10 Nov, 2 h)

**Goal.** A reusable training and evaluation loop whose metrics are comparable with notebook 02.

**Tasks**
1. `src/fundus/training.py`: `train_one_epoch`, `evaluate`, device selection (`mps`, then `cuda`, then `cpu`), `set_seed`, and saving predictions (image ID, true grade, predicted grade, class probabilities) to CSV.
2. Use `shortcut_metrics` from `fundus.features.shortcut` (rename later if you like) so that QWK, macro-F1 and balanced accuracy are computed exactly as before.
3. Notebook 04, part C: train `SmallCNN` for a few epochs on a 512-image subset at 128 px; plot loss and QWK per epoch; log images per second.

**Notebook 04, part C**
10. Training on a small subset
11. Learning curves
12. Throughput on this machine (images per second) and what it implies for the experiment budget
13. Summary: what part A to C taught me

**Audit**
- Two runs with the same seed give nearly identical losses. Note any non-determinism on MPS.
- Validation metrics come from the same function as notebook 02.
- Only train and val data are loaded.

**Expected result.** Loss decreases; validation QWK clearly above 0 and well below the metadata baseline (0.69). A small CNN on a small subset is not supposed to be good; the goal is a pipeline you can trust.

**Update docs.** LOG (throughput number).

**Done when.** Loop works; learning curves look sane; throughput logged.

### T-07 Notebook 05: transfer-learning baseline (Wed 11 and Thu 12 Nov, 4 h)

**Goal.** The first honest CNN number on the frozen split.

**Learn first (30 min).** PyTorch tutorial "Transfer Learning for Computer Vision"; freezing versus fine-tuning; class-weighted loss versus weighted sampling.

**Tasks**
1. `configs/baseline_resnet18.yaml`: size 256, batch 32, learning rates, epochs, seed.
2. Notebook `05_baseline_transfer_learning.ipynb`.
3. Save `reports/aptos_baseline_resnet18_summary.json` and `reports/preds/aptos_resnet18_val_predictions.csv`.

**Notebook 05 headings**
1. Purpose, inputs, outputs, rules (test sealed; validation used for epoch selection only)
2. Setup and configuration
3. Data loaders (cached 256 px, flips and rotations, ImageNet normalisation)
4. Model: ResNet18, ImageNet weights, new 5-class head
5. Training: 2 epochs with the backbone frozen, then fine-tune everything with a lower learning rate; class-weighted cross-entropy; keep the epoch with the best validation QWK
6. Evaluation: QWK, macro-F1, balanced accuracy, per-grade recall, confusion matrix
7. Comparison with the metadata references (validation QWK 0.692 for 13 features; 0.594 for the resolution lookup)
8. Observations and decisions

**Audit**
- State clearly that validation is used to pick the epoch, so the validation score is mildly optimistic. The unbiased number comes later from cross-validation or the test split.
- Prediction file IDs equal the validation IDs in the manifest.
- Record training time and images per second.
- Single seed in Sprint 1.

**Expected result.** The CNN beats the metadata baseline. If it does not, suspect a pipeline bug. If the random-split QWK is very high (around 0.9 or more), do not celebrate; T-08 shows how much comes from the source.

**Update docs.** LOG; README results table (new row "ResNet18, random split"); DECISIONS (loss and sampling choice).

**Done when.** Summary JSON and predictions file saved; numbers in text match.

### T-08 Notebook 06: per-source evaluation (Fri 13 Nov, 2 h)

**Goal.** Find out how much of the CNN score survives when the source is controlled.

**Notebook `06_per_source_evaluation.ipynb` headings**
1. Purpose
2. Load validation predictions from T-07
3. Metrics per resolution group (accuracy and confusion; QWK is degenerate inside a single-grade source, say so)
4. Source-matched subset (2588 x 1958 and 2416 x 1736): QWK of the T-07 model on matched validation images versus all validation images, with bootstrap 95% intervals over images
5. Exploratory: train a second model on matched training images only; compare on the same matched validation images
6. Comparison with the metadata-only model inside the matched sources (notebook 02, section 6)
7. Conclusions: the gap, and what it means for how results are reported

**Audit**
- Counts per group add up to the full validation set.
- The same hyperparameters are used for the matched-only model; note it is exploratory (about 960 images in the matched sources in total).
- Bootstrap uses at least 1,000 resamples and a fixed seed.

**Expected result.** The matched-subset score is lower than the all-data score; the size of the gap is the finding. Report uncertainty.

**Update docs.** README results table; DECISIONS D-007 to Accepted or Rejected with the evidence; LOG.

**Done when.** Gap and intervals reported in the notebook and the README.

### T-09 Sprint review (Sat 14 Nov, 2 h)

**Tasks**
1. Update the README status table and results table (random split, per source, matched subset, metadata baselines).
2. Fill `docs/SPRINT1_SUMMARY.md` (one page for your clinical collaborator).
3. Complete LOG and DECISIONS (D-007, D-009 decided).
4. Run the audit checklist in `docs/AUDIT_CHECKLIST.md`, section D.
5. Tag `v0.1-baseline` and push.

**Done when.** README numbers equal the JSON files; tag pushed.

### Buffer (none left)

Both buffer days were used (T-02 took 5 and 6 Nov). Sprint 1 now ends on Sat 14 Nov. Any further slip is absorbed only by the scope cuts listed at the top, never by longer nights. Meeting with the clinical collaborator (`docs/QUESTIONS_FOR_CLINICIAN.md`) should happen this week; do not wait for the sprint review.

Overflow from earlier tickets; meeting with your clinical collaborator using `docs/QUESTIONS_FOR_CLINICIAN.md` and `docs/SPRINT1_SUMMARY.md`; repository hygiene. If nothing overflowed, start T-10 early.

---

## Sprint 2 outline (15 to 28 Nov)

| Ticket | What | Deliverable |
|---|---|---|
| T-10 | Write the analysis plan **before** running anything: question (does CLAHE or local-contrast enhancement help grading?), variants, metric, seeds, splits, comparison rule | `docs/ANALYSIS_PLAN_enhancement.md` |
| T-11 | Config-driven training script with logging and saved predictions | `scripts/train.py`, `configs/*.yaml` |
| T-12 | Imbalance and augmentation study (class weights versus sampler; brightness and scale jitter), applied only after splitting; three seeds for the baseline | `notebooks/07_training_choices.ipynb` |
| T-13 | Enhancement ablation: 3 variants x 3 seeds, grouped folds, confidence intervals, per-source results | `notebooks/08_enhancement_ablation.ipynb` |
| T-14 | Sprint review | `docs/SPRINT2_SUMMARY.md` |

## Where each file goes

| File | Ticket |
|---|---|
| `src/fundus/preprocessing.py`, `tests/test_preprocessing.py` | T-02 |
| `scripts/cache_preprocessed.py` | T-03 |
| `notebooks/03_preprocessing_design.ipynb` | T-02, T-03 |
| `notebooks/04_cnn_fundamentals.ipynb` | T-04 to T-06 |
| `src/fundus/datasets/torch_dataset.py`, `tests/test_torch_dataset.py` | T-05 |
| `src/fundus/models/cnn.py` | T-05 |
| `src/fundus/training.py` | T-06 |
| `configs/baseline_resnet18.yaml`, `notebooks/05_baseline_transfer_learning.ipynb` | T-07 |
| `reports/aptos_baseline_resnet18_summary.json`, `reports/preds/*.csv` | T-07 |
| `notebooks/06_per_source_evaluation.ipynb` | T-08 |
| `docs/SPRINT1_SUMMARY.md` | T-09 |

## Risks

| Risk | Mitigation |
|---|---|
| Another interruption or slip | No buffer is left; use the scope cuts, not longer nights |
| Training is slow on MPS | Start at 128 or 256 px; subsets for debugging; measure images per second first |
| Baseline below the metadata reference | Treat as a pipeline bug; repeat the overfit test |
| Baseline looks suspiciously good | Check per-source results and duplicates first |
| Code gets ahead of understanding | Write each DECISIONS entry yourself; if you cannot explain a function, rewrite it by hand |
| Test partition used by accident | Dataset class refuses `test` unless `allow_test=True`; the test split is used once, at the end of a phase |

