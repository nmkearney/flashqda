# mapper.py

import pandas as pd
from typing import Dict, List, Optional

def assign_categories_to_dataframe(
    df: pd.DataFrame,
    column_names: List[str],
    clusters: Dict[int, List[str]],
    labels: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """
    For each text column in column_names, create a new `<col>_category` column
    that maps each cell's text to its cluster label.

    clusters: {cluster_id: [item_text, ...]}
    labels:   {cluster_id: "Readable theme name"} (optional)
    """
    # Build reverse lookup: item_text -> label string
    item_to_label: Dict[str, str] = {}

    for cid, items in clusters.items():
        label = labels.get(cid, f"Category_{cid}") if labels else f"Category_{cid}"
        for item in items:
            item_to_label[item] = label

    new_df = df.copy()
    for col in column_names:
        new_df[f"{col}_category"] = new_df[col].map(item_to_label)

    return new_df
