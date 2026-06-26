# mapper.py

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from flashqda.log_utils import update_log

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


def remap_from_categories_csv(
    original_file: Path,
    categories_file: Path,
    column_names: List[str],
    output_directory: Path,
    save_name: str = "grouped_items",
    log_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Re-apply a manually edited _categories.csv to the original dataframe.

    Reads the item-to-label mapping from categories_file (the Items column,
    semicolon-separated) and assigns a new <col>_category column for each
    column in column_names.

    Args:
        original_file:
            Path to the original input CSV (same file used in group_items).
        categories_file:
            Path to the edited *_categories.csv produced by group_items.
        column_names:
            Text columns to remap (must match the columns used originally).
        output_directory:
            Directory where the augmented CSV will be written.
        save_name:
            Base name for the output file.
        log_path:
            Optional log file path.

    Returns:
        Augmented DataFrame with updated <col>_category columns.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        if log_path:
            update_log(log_path, msg)

    log("[INFO] === Starting remap from edited categories ===")
    log(f"[INFO] Original file: {original_file}")
    log(f"[INFO] Categories file: {categories_file}")

    # Build item-to-label lookup from the Items column ("; "-separated).
    cats = pd.read_csv(categories_file)
    item_to_label: Dict[str, str] = {}
    for _, row in cats.iterrows():
        label = str(row["Label"])
        for item in str(row["Items"]).split("; "):
            item = item.strip()
            if item:
                item_to_label[item] = label

    log(f"[INFO] Built mapping for {len(item_to_label)} items across {len(cats)} categories.")

    df = pd.read_csv(original_file)
    log(f"[INFO] Loaded original dataframe with {len(df)} rows.")

    df_aug = df.copy()
    for col in column_names:
        df_aug[f"{col}_category"] = df_aug[col].map(item_to_label)

    for col in column_names:
        n_unmapped = int(df_aug[f"{col}_category"].isna().sum())
        if n_unmapped:
            log(f"[WARN] {n_unmapped} rows in '{col}' had no match in the categories file.")

    out_path = output_directory / f"{save_name}_with_categories.csv"
    df_aug.to_csv(out_path, index=False)
    log(f"[INFO] Saved augmented dataframe to {out_path}")
    log("[INFO] === Remap complete ===")

    return df_aug
