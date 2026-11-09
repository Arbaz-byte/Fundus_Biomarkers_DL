"""Shared record and adapter interface for fundus image datasets.

Dataset-specific adapters convert their source files and annotations into a
common table schema. This lets downstream splitting, preprocessing, and
evaluation code work consistently across public and clinical datasets.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

REQUIRED_COLUMNS = ["path", "label", "patient_id", "site"]
"""Column names required in every normalized dataset table."""


@dataclass
class FundusRecord:
    """Represent the standard metadata for one fundus image.
    Args:
        path: Filesystem path to the image.
        label: Integer target label, or ``None`` when the label is unavailable.
        patient_id: Identifier used to associate images from the same patient
            and support patient-aware data handling.
        site: Dataset or acquisition-site identifier. Defaults to
            ``"unknown"``.
    """
    
    path: str
    label: Optional[int]
    patient_id: str
    site: str = "unknown"


class FundusDataset:
    """Base class for adapters that expose image data as a normalized table.
    
    Subclasses configure their source paths and implement ``build_table`` to
    produce a DataFrame containing at least the columns in
    ``REQUIRED_COLUMNS``. Call ``table`` to validate and return only those
    columns in their canonical order.
    """
    
    name = "base"
    
    def __init__(self, root: str):
        """Store the root directory for this dataset.
        
        Args:
            root: Filesystem path to the dataset's root directory. It is
                normalized to a ``pathlib.Path`` and available to subclasses
                as ``self.root``.
        """
        self.root = Path(root)

    def build_table(self) -> pd.DataFrame:
        """Build the dataset-specific table before schema normalization.
        
        Subclasses must override this method to read their source data and
        provide at least ``path``, ``label``, ``patient_id``, and ``site``.
        They may include additional columns; ``table`` removes those from its
        returned result.
        
        Returns:
            A DataFrame containing the required dataset columns.
            
        Raises:
            NotImplementedError: If a subclass does not provide an
                implementation.
        """
        raise NotImplementedError

    def table(self) -> pd.DataFrame:
        """Return the dataset as a validated, normalized DataFrame.
        
        This provides a consistent four-column interface to downstream
        processing, regardless of the source dataset's original schema.
        
        Returns:
            DataFrame containing ``path``, ``label``, ``patient_id``, and
            ``site`` in that order, with a fresh zero-based index. Any extra
            columns returned by ``build_table`` are omitted.
            
        Raises:
            NotImplementedError: If ``build_table`` is not implemented by a
                subclass.
            ValueError: If the built DataFrame is missing one or more
                required columns.
        """
        df = self.build_table()
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{self.name}: missing columns {missing}")
        return df[REQUIRED_COLUMNS].reset_index(drop=True)
