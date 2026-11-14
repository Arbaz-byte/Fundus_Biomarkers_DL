# Data policy

1. The repository may contain code, documentation, and dataset-derived material only when its terms permit publication.

2. Never commit clinical, institutional, or patient-derived data, images, metadata, or model outputs.

3. Keep all APTOS raw data, transformed images, caches, split manifests, and per-image measurements or predictions local and git-ignored.

4. The active source is the official [APTOS 2019 Kaggle competition download](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data), governed by its [competition rules](https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules), especially sections 7 and 8. See [DATASETS.md](DATASETS.md).

5. Section 7 permits non-commercial competition, academic research, and education use, but restricts making Competition Data available to people who have not agreed to the rules. Do not publish APTOS images or transformed image examples in this repository; Obtain explicit written permission before any public image display.

6. Do not publish per-image rows, image IDs with labels, split manifests, or caches. Check applicable terms and obtain clarification before publishing aggregate results, figures, model artifacts, or other derived outputs whose status is not explicit.

7. MIT covers this repository's code only, not APTOS data. Kaggle rule 8 has separate Competition Code sharing conditions; confirm whether its location requirement applies before relying on GitHub publication.

8. The active Kaggle download contains training and test partitions, not a validation partition. Create validation only from labeled training data. Keep the official test partition sealed until final evaluation.

9. Older analyses made from the MariaHerreroT third-party copy are superseded and must not be presented as findings on the official competition download. Do not mix the two sources.

10. Saved notebook outputs must not contain personal filesystem paths, APTOS images, or per-image data. Use synthetic or separately licensed examples in public demonstrations.
