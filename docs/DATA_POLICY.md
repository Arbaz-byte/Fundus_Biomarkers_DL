# Data policy

1. This repository holds code and public-data results only.
2. No clinical, institutional or patient-derived data, images, metadata or model outputs on such data are ever committed here.
3. If a clinical dataset is approved later, analysis on it happens in a separate private location under the institution's rules. Only aggregate metrics approved for release may be added here.
4. Public datasets are used under their own licences and are never redistributed here. This includes derived copies: do not commit dataset images, thumbnails, or notebook outputs that display them unless the licence clearly allows it.
5. Files in `reports/` contain aggregate results only. Prediction files may list public image IDs and predictions, nothing else.
6. Saved notebook outputs must not contain personal file-system paths.
7. The test partition of every dataset is sealed until the final evaluation of a phase and is never used for tuning.
