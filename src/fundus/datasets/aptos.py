"""APTOS 2019 Blindness Detection adapter.

Expected layout after downloading from Kaggle:

    data/raw/aptos/
        train.csv              columns: id_code, diagnosis
        train_images/<id_code>.png

Labels: 0 = No DR, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Proliferative.

Known limitations (state these in any write-up):
- APTOS has no patient identifier, so each image is treated as its own patient.
  Splits are therefore image-level and cannot rule out two images from one person.
- The public test set has no labels, so train.csv is split into train/val/test.
- Images come from several clinics and cameras with uneven quality.
"""
from pathlib import Path

import pandas as pd

from .base import FundusDataset


class AptosDataset(FundusDataset):
    name = "aptos"

    def __init__(self, root: str, image_dir: str = "train_images", csv_name: str = "train.csv"):
        super().__init__(root)
        self.image_dir = self.root / image_dir
        self.csv_path = self.root / csv_name

    def build_table(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Missing {self.csv_path}. See data/README.md and docs/DATASETS.md")
        df = pd.read_csv(self.csv_path)
        df["path"] = df["id_code"].map(lambda i: str(self.image_dir / f"{i}.png"))
        df["label"] = df["diagnosis"].astype(int)
        df["patient_id"] = df["id_code"].astype(str)  # no patient ID available
        df["site"] = "aptos"
        missing = [p for p in df["path"] if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} image files listed in the CSV are missing, e.g. {missing[0]}")
        return df
