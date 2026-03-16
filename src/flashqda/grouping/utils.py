# flashqda/grouping/utils.py
import pandas as pd

def extract_unique_items(df, column_names):
    """Return a unique list of non-null items from selected columns."""
    series_list = [df[col].dropna().astype(str) for col in column_names if col in df.columns]
    return pd.concat(series_list).drop_duplicates().tolist()

# utils.py (new helper or a new file dendrogram_utils.py if you prefer)

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np
from typing import List

def save_dendrogram(
    linkage_matrix: np.ndarray,
    labels: List[str],
    out_path,
    max_labels: int = 200,
):
    """
    Save a dendrogram plot to disk for audit / thematic map.

    We cap labels because very long dendrograms with hundreds of leaves
    become unreadable. For big corpora > max_labels, we still plot the
    structure but drop leaf labels.
    """
    plt.figure(figsize=(12, 6))

    if len(labels) <= max_labels:
        dendrogram(
            linkage_matrix,
            labels=labels,
            leaf_rotation=90,
            leaf_font_size=8,
        )
    else:
        dendrogram(
            linkage_matrix,
            no_labels=True,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

