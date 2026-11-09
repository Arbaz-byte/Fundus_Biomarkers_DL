"""Dataset adapter for APTOS 2019 Blindness Detection.

Expected layout after downloading from Kaggle:

    <root>/
        train.csv
        train_images/<id_code>.png

The CSV must contain ``id_code`` and ``diagnosis`` columns. The adapter maps
those provider-specific fields into the common fundus dataset table
(``path``, ``label``, ``patient_id``, ``site``), allowing downstream
preprocessing and evaluation code to work with APTOS through the same
interface as other datasets.

APTOS labels are diabetic-retinopathy grades: 0 = No DR, 1 = Mild,
2 = Moderate, 3 = Severe, 4 = Proliferative.

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
    """Adapt APTOS CSV rows and images to the common fundus dataset schema.
    Since the source dataset does not provide patient identifiers, ``id_code``
    is used as ``patient_id``. This makes the table compatible with the
    shared interface but does not enable patient-level split protection.
    """
    
    name = "APTOS-2019 "
    
    def __init__(self, root: str, image_dir: str = "train_images", csv_name: str = "train.csv"):
        """Configure paths to the APTOS image directory and label CSV.
        
        Args:
            root: Dataset root directory containing the CSV and image
                directory.
            image_dir: Image directory path relative to ``root``.
                Defaults to ``"train_images"``.
            csv_name: Label CSV path relative to ``root``. It must contain
                ``id_code`` and ``diagnosis`` columns. Defaults to
                ``"train.csv"``.
        """
        super().__init__(root)
        self.image_dir = self.root / image_dir
        self.csv_path = self.root / csv_name
        
    def build_table(self) -> pd.DataFrame:
        """Read the label CSV and return one validated row per listed image.
        
        The source columns are preserved, with these adapter columns added:
        ``path`` (constructed as ``image_dir / <id_code>.png``), ``label``
        (``diagnosis`` converted to an integer), ``patient_id`` (the string
        form of ``id_code``), and ``site`` (the constant ``"aptos"``). Using
        the shared columns lets downstream code process APTOS consistently
        with other dataset adapters.
        
        Returns:
            DataFrame with the original CSV data and the four adapter columns.
            Every generated image path is checked before the table is returned.
            
        Raises:
            FileNotFoundError: If the configured CSV is absent or any image
                listed by the CSV is missing.
            KeyError: If the CSV does not contain ``id_code`` or
                ``diagnosis``.
            ValueError: If a diagnosis value cannot be converted to an
                integer.
        """
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
