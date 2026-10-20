"""Common dataset interface.

Every dataset (public or clinical) is wrapped by an adapter that produces a table with
the same columns. Real clinical data later plugs in as one more adapter.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

REQUIRED_COLUMNS = ["path", "label", "patient_id", "site"]


@dataclass
class FundusRecord:
    path: str
    label: Optional[int]
    patient_id: str
    site: str = "unknown"


class FundusDataset:
    """Base adapter. Subclasses implement build_table()."""

    name = "base"

    def __init__(self, root: str):
        self.root = Path(root)

    def build_table(self) -> pd.DataFrame:
        raise NotImplementedError

    def table(self) -> pd.DataFrame:
        df = self.build_table()
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{self.name}: missing columns {missing}")
        return df[REQUIRED_COLUMNS].reset_index(drop=True)
