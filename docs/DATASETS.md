# Datasets

The code licence does not cover dataset contents.

## Active dataset

| Dataset | Purpose | Source | Data-use terms | Downloaded | Local path |
|---|---|---|---|---|---|
| APTOS 2019 Blindness Detection competition data | DR grading and shortcut audit | [Kaggle competition data page](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) | [2019 competition rules](https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules), especially sections 7 and 8; see summary below | 14 November 2026 | `data/raw/APTOS-2019-kaggle-competition/` |

The downloaded package is organized as `train_images/` with `train.csv`, and `test_images/` with `test.csv`. It has no provided validation split. The official training CSV has `id_code` and `diagnosis`; the competition test CSV has `id_code` only. Create validation and any local held-out test partition from the labeled official training data.

### Provenance

The competition was organized for Aravind Eye Hospital & PG Institute of Ophthalmology in Madurai, India, together with the Asia Pacific Tele-Ophthalmology Society (APTOS). This does not establish that every image was captured at one hospital or clinic. The Kaggle competition package is the project's active source; third-party copies and their labels are not used to establish its terms.

### Competition rules and project handling

The following is a project summary of the accepted Kaggle competition rules, not legal advice. The [full rules](https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules) govern.

- **Section 7.A — Data Access and Use:** use is limited to non-commercial purposes, including participating in the competition, academic research, and education.
- **Section 7.B — Data Security:** take reasonable measures to prevent access by people who have not agreed to the rules. The rules prohibit transmitting, duplicating, publishing, redistributing, or otherwise making Competition Data available to a party not participating in the Competition.
- The rules state no small-sample exception. Do not put APTOS source pixels, transformed image examples, image-bearing notebook outputs, or image IDs paired with labels in this public repository. There is no 10-image allowance; do not publish even a few examples without separate written permission that clearly authorizes public display.
- Keep per-image measurements, predictions, split manifests, and caches local and git-ignored. Keep the official test set sealed; create any validation split from the labeled training set only.
- Aggregate metrics and figures are not automatically cleared by the above. Check the rules and obtain clarification before publishing dataset-derived results, model artifacts, or other outputs whose sharing status is not explicit.
- **Section 8 — Submission Code Requirements** is separate from permission to use or redistribute data. It describes conditions for sharing Competition Code, including a Kaggle forum/kernel location and an OSI-approved licence that does not restrict commercial use. This project's code is MIT-licensed, but whether section 8's location condition applies to this independent, post-competition GitHub project should be confirmed before treating GitHub as sufficient.

### Previous source — superseded

The earlier copy at `data/raw/APTOS-2019/` came from the third-party [MariaHerreroT Kaggle dataset](https://www.kaggle.com/datasets/mariaherrerot/aptos2019). That upload did not provide a verified licence in this project record. It had a pre-split `train_images`, `val_images`, and `test_images` layout with `train_1.csv`, `valid.csv`, and `test.csv`. It is no longer an active project source.

The local per-ID report at `data/processed/aptos_source_comparison.csv` confirms that the IDs, labels, and exact file-byte SHA-256 hashes in the old copy's three labeled folders match the official competition training set. The old copy is therefore not an independent dataset, and its folder named `test` is not the Kaggle competition test set. This content comparison does not verify the old uploader's provenance or licence.

Old counts, split assignments, shortcut scores, preprocessing QA, cache outputs, and model results remain **historical**, not accepted results for the active workflow. Exact file identity means the old copy does not add different image pixels, but previous split-dependent results still need to be reproduced from the official root using the new, controlled split. Do not combine source folders. Keep the old folder git-ignored and remove it only after confirming it is no longer needed.

The audit script and notebook write per-image metadata and derived tables only to git-ignored `data/processed/`. Keep the per-ID comparison report and audit outputs local; do not publish their contents or aggregate findings until the applicable competition terms have been checked.

## Other datasets considered

| Dataset | Intended use | Status |
|---|---|---|
| DRIVE, CHASE_DB1, STARE | Vessel segmentation | Planned for Phase 3 |
| IDRiD | Lesions, Indian data | Planned for Phase 5 |
| ODIR-5K, Messidor-2, REFUGE | Multi-disease, external DR test, disc/cup | Later |
