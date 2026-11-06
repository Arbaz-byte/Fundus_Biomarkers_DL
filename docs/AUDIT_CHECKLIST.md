# Audit checklist

Run the relevant section before you trust, report or commit a number. Tick the box in the ticket's LOG entry, not here.

## A. Data and split
- [ ] The partition of every image comes from `configs/splits/aptos_v1.csv` only.
- [ ] Train, validation and test IDs are disjoint (assert in code).
- [ ] Near-duplicates are grouped across **all** images before partitions are assigned; number of groups spanning train and validation is recorded.
- [ ] Test images are never opened, cached or profiled before the final evaluation of a phase.
- [ ] Class counts per partition printed and compared with the manifest.
- [ ] Augmentation happens after splitting and on training data only.

## B. Code
- [ ] Reusable logic lives in `src/fundus/` with a test; notebooks only call it.
- [ ] No hard-coded absolute paths; `rel(...)` used when printing.
- [ ] Seeds set and logged.
- [ ] One module per purpose (no duplicate files such as `preprocess.py` and `preprocessing.py`).
- [ ] Same metric function everywhere (QWK, macro-F1, balanced accuracy).
- [ ] Preprocessing is identical for train, validation and any later split.

## C. Results
- [ ] A baseline sits next to every model (majority class, metadata-only, resolution lookup).
- [ ] Uncertainty reported (fold spread, seeds, or bootstrap interval).
- [ ] Results reported per source (resolution group) as well as overall.
- [ ] A score that looks too good triggers a leakage hunt before anything else.
- [ ] Validation set use is stated (what it was used for and how often).
- [ ] Numbers in text equal the printed output and the saved JSON/CSV files.
- [ ] Every claim in the docs (for example `docs/PREPROCESSING.md`) has a number or a reproducible demonstration.

## D. Repository hygiene
- [ ] `pytest -q` is green.
- [ ] Every notebook runs top to bottom after Restart kernel.
- [ ] No personal paths (`/Users/`) and no dataset images in saved outputs unless the licence allows them.
- [ ] `docs/LOG.md`, `docs/DECISIONS.md`, README status and results tables updated.
- [ ] `reports/` contains aggregate files only.
- [ ] `git status` clean except intended files; commit messages short and specific.
