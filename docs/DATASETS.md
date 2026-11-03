# Datasets

Read and respect every licence. Fill in the bracketed fields yourself; they are the facts only you can confirm.

## In use

| Dataset | Use | Source | Licence / terms | Date downloaded | Notes |
|---|---|---|---|---|---|
| APTOS-2019 (pre-split download) | DR grading, shortcut audit | [Kaggle dataset URL of the exact upload you used] | [copy the licence/competition rules text or link] | [date] | Folder layout: `train_images`, `val_images`, `test_images`, `train_1.csv`, `valid.csv`, `test.csv`. The provider split is **not** used; our split is `configs/splits/aptos_v1.csv`. |

### Facts observed in the APTOS-2019 download (development set: train + validation images)

- 2,963 images: 2,438 train, 525 validation. Grades 0 to 4: 1,526 / 287 / 783 / 143 / 224.
- 17 distinct resolutions, from 474 x 358 up to 4288 x 2848.
- Resolution is strongly linked to the grade (notebook 01, section 5). For example, 2048 x 1536 is 100% No DR, 3216 x 2136 is 0.6% No DR.
- No patient identifiers.

### Open questions about this dataset

- [ ] Who produced the pre-split upload, and how were its three partitions made?
- [ ] Which clinics/cameras do the different resolutions correspond to? (Ask the clinical collaborator; check the competition documentation.)
- [ ] Are redistribution and derived thumbnails allowed? (Decides whether sample images may appear in notebook outputs.)

## Considered, not downloaded

| Dataset | Intended use | Status |
|---|---|---|
| Resized 2015 + 2019 DR Detection (Kaggle, uploader c7934597; about 18.5 GB) | Cross-dataset test, train on 2015 and test on 2019 | Not downloaded. Contents, folder structure and licence not verified. Planned for Phase 5. |
| DRIVE, CHASE_DB1, STARE | Vessel segmentation | Phase 3 |
| IDRiD | Lesions, Indian data | Phase 5 |
| ODIR-5K, Messidor-2, REFUGE | Multi-disease, external DR test, disc/cup | later |

For each dataset, before first use: source link, licence, download date, and what the files contain.

Rule: never redistribute datasets; cite each dataset in any write-up.
