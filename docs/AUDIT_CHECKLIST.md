# Audit checklist

Run the relevant section before trust anything , report or commit a number. Tick the box in the ticket's LOG entry, not here.

## A. Data and split
- [ ] Active input is `data/raw/APTOS-2019-kaggle-competition/`; no files from the superseded third-party copy are mixed in.
- [ ] Run notebook 00 / `scripts/audit.py`; review CSV schemas, source reconciliation, class balance, and resolution-by-grade outputs.
- [ ] Training CSV columns, row count, labels, image count, and missing/orphaned files are audited and recorded.
- [ ] Validation is derived only from labeled training data; the official test partition is sealed and not opened, cached, or profiled before final evaluation.
- [ ] Train and validation IDs are disjoint; near-duplicates are grouped before assigning them to partitions.
- [ ] Class counts per local partition are printed and compared with the manifest.
- [ ] Augmentation happens after splitting and on training data only.

## B. Code
- [ ] Reusable logic lives in `src/fundus/` with a test; notebooks only call it.
- [ ] No hard-coded absolute paths; `rel(...)` used when printing.
- [ ] Seeds set and logged.
- [ ] One module per purpose (no duplicate files such as `preprocess.py` and `preprocessing.py`).
- [ ] Same metric function everywhere (QWK, macro-F1, balanced accuracy).
- [ ] Preprocessing is identical for train, validation and any later split.
- [ ] Notebooks run from a clean kernel: no cell depends on code that was deleted or moved.
- [ ] When the "largest" or "smallest" items are selected, check for ties (a top-6 of a tied value is an arbitrary subset); count the whole group.

## C. Results
- [ ] A baseline sits next to every model (majority class, metadata-only, resolution lookup).
- [ ] Uncertainty reported (fold spread, seeds, or bootstrap interval).
- [ ] Results reported per source (resolution group) as well as overall.
- [ ] A score that looks too good triggers a leakage hunt before anything else.
- [ ] Validation set use is stated (what it was used for and how often).
- [ ] Numbers in text equal the printed output and the saved JSON/CSV files.
- [ ] Every current preprocessing claim in `docs/APTOS_PREPROCESSING.md` has a number or reproducible demonstration; `docs/PREPROCESSING.md` is historical only.

## D. Repository hygiene
- [ ] `pytest -q` is green.
- [ ] Every notebook runs top to bottom after Restart kernel.
- [ ] No personal paths (`/Users/`) or dataset images in saved notebook outputs; clear image-bearing output cells before committing.
- [ ] Check `git check-ignore -v` for raw/processed data and confirm only the intended `.gitkeep` markers are unignored.
- [ ] No APTOS IDs with labels, per-image results, or local split manifest are staged.
- [ ] No APTOS image or transformed example is shared publicly based on a sample-count assumption; the rules provide no 10-image exception.
- [ ] Before sharing Competition Code publicly, check Kaggle rule 8's forum/kernel location and OSI/commercial-use conditions.
- [ ] `docs/LOG.md`, `docs/DECISIONS.md`, README status and results tables updated.
- [ ] `reports/` contains aggregate files only.
- [ ] `git status` clean except intended files; commit messages short and specific.

