# flashqda/grouping/utils.py
import pandas as pd

def extract_unique_items(df, column_names):
    series_list = []
    for col in column_names:
        if col in df.columns:
            s = df[col].dropna().astype(str).str.strip()
            s = s[s != ""]
            series_list.append(s)

    if not series_list:
        return []

    return pd.concat(series_list).drop_duplicates().tolist()

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
    try:
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram
    except ImportError as e:
        raise ImportError(
            "matplotlib and scipy are required for dendrogram plotting. "
            "Install them with 'pip install matplotlib scipy' or set "
            "'save_dendrogram_plot=False' in group_items."
    ) from e

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

