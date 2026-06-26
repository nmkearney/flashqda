from __future__ import annotations

import pandas as pd
from typing import List


def validate_columns(df: pd.DataFrame, required_cols: List[str], context: str = "") -> None:
    """
    Raise ValueError if any required columns are absent from df.
    Provides a clear message listing what is missing and what was found.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        prefix = f"[{context}] " if context else ""
        raise ValueError(
            f"{prefix}Input CSV is missing required columns: {missing}. "
            f"Columns found: {list(df.columns)}"
        )
