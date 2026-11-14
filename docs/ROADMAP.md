# Roadmap (v5, 2026-11-14)

This roadmap replaces all earlier versions. It starts from the official-source reset of 14 November 2026: the project now uses only the official APTOS 2019 Kaggle competition download (see [`DATASETS.md`](DATASETS.md) and [`DATA_POLICY.md`](DATA_POLICY.md)). Results from the earlier third-party copy are historical (see [`LOG.md`](LOG.md)) and are not acceptance targets.

**Deadline for the core work: 31 January 2027.** February 2027: deploy an interactive demo and portfolio site, and prepare a proof-of-work page for the clinical collaborators. The demo may not display APTOS images (see the data policy); it will use user uploads or separately licensed or synthetic examples.

## Status at the start of this roadmap

- **Day 00 (14 Nov) done:** the official package is downloaded; an audit reconciled the earlier labelled folders with the official training data and profiled class balance and resolution by grade. The official `test.csv` contains IDs only (no labels), and no test image file was opened. Detailed audit outputs stay local.

- **Consequence:** the official test partition cannot be used for labelled evaluation. The project therefore creates a **local three-way split** (train, validation, local test) from the labelled training set. The local test partition stays sealed until a phase's final evaluation. See decision D-005 (to record).

- **Next:** repository hygiene, split, metadata and shortcut re-analysis, preprocessing re-audit, cache, baseline.

## Rhythm

- **Block days (Tue, Thu, Sat):** morning about 3 h of active learning (derive, code from scratch, explain), evening 1 to 2 h on the project.

- **Sunday (half day):** flex: tidy code and docs, README, LOG, review, rest.

- One new concept per block. Each block has a **Core** (about 70%, must do) and a **Stretch** (optional). A missed block is moved, not doubled.
- Hours: about 15 to 17 h a week, 11 weeks to 31 Jan, roughly 165 to 185 h in total. The plan uses about 150 h; the rest is the only buffer.

## Rules that apply to every ticket

1. **Provenance first.** Every dataset starts with a recorded source, terms and download date in [`DATASETS.md`](DATASETS.md) before any analysis. This includes any external or vessel-segmentation dataset added later.

2. **Local only.** Image-bearing outputs, per-image rows, image IDs with labels, split manifests and caches stay in git-ignored locations. Only aggregate results go to `reports/`.

3. **Sealed local test.** Used once, at the end of a phase, never for tuning.

4. **Audit before trust.** Use [`AUDIT_CHECKLIST.md`](AUDIT_CHECKLIST.md); numbers in text equal the printed output.

5. **Reusable code lives in `src/fundus/` with a test;** notebooks call it and interpret. Notebooks run from a clean kernel. Experiments are kept in `NNb_<topic>` notebooks, never deleted.

6. **Record:** LOG every working day; DECISIONS when a choice is made, with the reason in the author's own words.

## Calendar

| Sprint | Dates | Theme | Core deliverable |
|---|---|---|---|
| 1 | Sun 15 Nov to Sun 6 Dec | Restart and foundations | local split, re-run metadata and shortcut analyses, preprocessing re-audit, cache, baseline CNN, per-source evaluation |
| 2 | Mon 7 Dec to Sun 20 Dec | Honest evaluation | training choices (imbalance, augmentation), repeated seeds, one local-test evaluation of the chosen baseline, meeting with clinical collaborators |
| 3 | Mon 21 Dec to Sun 3 Jan | Vessel segmentation (holiday weeks: two block days per week) | segmentation model on a vessel dataset with recorded terms |
| 4 | Mon 4 Jan to Sun 17 Jan | Vessel biomarkers and statistics | tortuosity, vessel density, fractal dimension; comparison across disease groups; uncertainty |
| 5 | Mon 18 Jan to Sun 31 Jan | Packaging | README, report, figures, one-page proof of work, code cleanup and tests, tag `v1.0` |
| Feb | February 2027 | Deploy | interactive demo, portfolio site |

**Stretch, only if ahead:** enhancement experiment (CLAHE or local-average subtraction versus raw, several seeds); cross-dataset study (needs an officially obtained or separately licensed second dataset with terms recorded; a third-party re-upload of competition data is not acceptable); artery/vein ratio; foundation-model fine-tuning.

**First levers if time runs short:** reduce the number of biomarkers, drop repeated seeds to two, shorten Sprint 2 to one week.

## Sprint 1: restart and foundations (15 Nov to 6 Dec)

| Date | Morning: theory (about 3 h) | Evening: project (1 to 2 h) | Done when |
|---|---|---|---|
| Sun 15 Nov (half) | none | **Repository hygiene:** check git history for committed labelled IDs, image outputs and the old split manifest, and decide on a history rewrite; fix links to files that are no longer public; renumber decisions; record the Day 00 audit summary and decisions D-005 to D-007; push this roadmap | history check result in LOG; `git check-ignore -v` confirms raw data, processed data and manifests are ignored; no dead links |
| Tue 17 Nov | Scores, loss, backpropagation, autograd (derive on paper, code in NumPy, check the gradient numerically, compare with PyTorch) | **Split:** stratified train / validation / local-test split from the labelled training set; near-duplicates grouped first; manifest local and git-ignored; rewrite the split script for the flat official layout | partitions disjoint; class counts printed; no duplicate group crosses a partition; tests pass |
| Thu 19 Nov | Convolution: filters, feature maps, pooling, output size, parameter count (NumPy `conv2d` against PyTorch) | **Metadata profiling:** rerun notebook 01 on the official training images (train and validation only); adapt paths to the flat layout | metadata table complete and matching the manifest; no per-image outputs staged |
| Sat 21 Nov | `nn.Module`, `Dataset`, `DataLoader` | **Shortcut test:** rerun notebook 02 on the new train/validation split; compare with the archived old-copy numbers and record whether the images were identical | numbers recorded; text matches output |
| Sun 22 Nov (half) | none | **Flex:** README, LOG, DECISIONS; weekly review; teach-back of one concept | docs updated |
| Tue 24 Nov | Optimization practice: learning rate, optimizers, regularisation, overfitting, device (`mps`) | **Preprocessing re-audit:** mask QA sample across resolution families, saturated-mask count, geometry checks; fill measured results into `APTOS_PREPROCESSING.md` | QA verdicts and saturation rate recorded; every claim has a number |
| Thu 26 Nov | Architectures and transfer learning (residual connections, pretrained networks) | **Cache and cue test:** rebuild the 256 px cache for train and validation; rerun notebook 03c | cache verified; cue-test numbers recorded |
| Sat 28 Nov | Consolidation (review the earlier gates; no new concept) | **Build:** `FundusTorchDataset`, `SmallCNN`, overfit-one-batch test | tests pass; 32 images can be memorised |
| Sun 29 Nov (half) | none | **Flex** | docs updated |
| Tue 1 Dec | none new | **Training loop and baseline run 1:** loop module, small run, frozen-backbone ResNet18 | curves and per-grade recall observed; what would be suspicious written down before looking |
| Thu 3 Dec | none new | **Baseline run 2:** fine-tune; evaluate against metadata baselines recomputed on the new split | aggregate metrics saved; per-image predictions local only |
| Sat 5 Dec | none new | **Per-source evaluation:** source-matched subset defined from the new audit; bootstrap intervals | gap and intervals reported |
| Sun 6 Dec (half) | none | **Sprint 1 review:** README results (aggregate only), LOG, DECISIONS; tag `v0.1-baseline` if everything staged is publishable | tag pushed |

Dates move if a block is missed; the sprint review can move by a few days. If a morning runs over, split the lecture across two blocks and move the build part.

## Sprints 2 to 5 (outline; each gets block-by-block detail when it starts)

- **Sprint 2 (7 to 20 Dec):** class weights versus sampler; flips and rotations (further augmentations only if justified); three seeds for the baseline; one evaluation on the local test partition for the chosen baseline; meeting with clinical collaborators (share notebooks, not images: they download the competition data themselves under its rules); sprint review.
- **Sprint 3 (21 Dec to 3 Jan):** theory first (segmentation, U-Net, Dice loss); intake of a vessel dataset (provenance and terms recorded); segmentation model; qualitative checks. Vessel masks computed on APTOS images stay local.
- **Sprint 4 (4 to 17 Jan):** skeletonisation and vessel features (tortuosity, density, fractal dimension), statistics across disease groups, honest uncertainty; aggregate results only.
- **Sprint 5 (18 to 31 Jan):** README, report, figures without APTOS pixels, proof-of-work page, code cleanup and tests, tag `v1.0`.
- **February:** plan and build the interactive demo and portfolio site; no APTOS images in the demo.

## Where files go (Sprint 1)

| File | Block |
|---|---|
| `scripts/audit.py`, `notebooks/00_*` (Day 00 audit; outputs local) | done |
| `scripts/create_aptos_split.py` (rewritten), local manifest under `data/processed/` (git-ignored) | Tue 17 Nov |
| `notebooks/01_image_variation.ipynb` (rerun) | Thu 19 Nov |
| `notebooks/02_metadata_shortcut_test.ipynb` (rerun) | Sat 21 Nov |
| `notebooks/03_preprocessing_design.ipynb`, `docs/APTOS_PREPROCESSING.md` | Tue 24 Nov |
| `scripts/cache_preprocessed.py`, `notebooks/03c_cache_and_cue_test.ipynb` | Thu 26 Nov |
| `src/fundus/datasets/torch_dataset.py`, `src/fundus/models/cnn.py`, `tests/` | Sat 28 Nov |
| `src/fundus/training.py`, `notebooks/05_baseline_transfer_learning.ipynb`, `reports/` aggregate summary | 1 to 3 Dec |
| `notebooks/06_per_source_evaluation.ipynb` | Sat 5 Dec |

## Risks

| Risk | Response |
|---|---|
| The reset consumed a week | Stretch items stay out; Sprint 2 can shrink; the levers above |
| Old scripts assume the old folder layout | Read each script before running it; update paths and tests first |
| Theory takes longer than planned | Split a lecture over two blocks and move the build part |
| Blocks lost to other commitments | Move the content; do not double up |
| A score looks too good | Per-source results and a leakage search before any interpretation |
| Local test partition used by accident | The dataset class refuses it unless `allow_test=True`; used once per phase |
| Publishing something the data terms do not allow | Aggregate outputs only; check [`DATASETS.md`](DATASETS.md) and the audit checklist before every push |
