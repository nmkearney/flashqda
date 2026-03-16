from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from flashqda.embeddings.provider import embed_texts
from flashqda.grouping.clusterer import hierarchical_cluster_items
from flashqda.grouping.labeller import label_clusters
from flashqda.grouping.mapper import assign_categories_to_dataframe
from flashqda.grouping.utils import extract_unique_items, save_dendrogram
from flashqda.log_utils import update_log


def group_items(
    input_file: Path,
    column_names: List[str],
    output_directory: Path,
    embedding_model: str = "text-embedding-3-large",
    similarity_threshold: float = 0.6,
    linkage_method: str = "average",
    use_llm_labels: bool = True,
    llm_model: str = "gpt-4o",
    save_name: str = "grouped_items",
    # infra / safety knobs
    cache_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
    label_checkpoint_path: Optional[Path] = None,
):
    """
    Main public API: run AHC-based semantic grouping workflow on a CSV file.

    Args:
        input_file:
            Path to the CSV file containing the dataset
            (e.g. with 'cause'/'effect' columns).
        column_names:
            List of text columns whose values will be embedded and grouped.
            Example: ["cause", "effect"].
        output_directory:
            Directory where outputs will be written.
        embedding_model:
            Embedding model (e.g. 'text-embedding-3-small').
        similarity_threshold:
            Cosine similarity cutoff (0–1). Higher = tighter, more granular clusters.
        linkage_method:
            Linkage rule for AHC. "average" is usually best for cosine distance.
        use_llm_labels:
            If True, generate human-readable cluster labels using the LLM.
            If False, clusters will just be "Category_<id>".
        llm_model:
            Model name to use for labeling clusters (passed to label_clusters).
        save_name:
            Base name / prefix for all exported artifacts.
        cache_path:
            Optional path for the embeddings cache JSON.
            If None, defaults to output_directory / f"{save_name}_embeddings.json"
        log_path:
            Optional log file path. If None, defaults to output_directory / f"{save_name}_log.txt"
        label_checkpoint_path:
            Optional path to JSON checkpoint for generated labels.
            If None, defaults to output_directory / f"{save_name}_label_checkpoint.json"

    Returns:
        df_aug:
            Augmented pandas DataFrame with new `<col>_category` columns.
            (The same dataframe is also written to disk.)
    """

    # ---------------------------------------------------------------------
    # Resolve/prepare paths
    # ---------------------------------------------------------------------
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    if cache_path is None:
        cache_path = output_directory / f"{save_name}_embeddings.json"

    if log_path is None:
        log_path = output_directory / f"{save_name}_log.txt"

    if label_checkpoint_path is None:
        label_checkpoint_path = output_directory / f"{save_name}_label_checkpoint.json"

    update_log(log_path, f"[INFO] === Starting grouping pipeline (AHC) ===")
    update_log(log_path, f"[INFO] Input file: {input_file}")
    update_log(log_path, f"[INFO] Columns to group: {column_names}")
    update_log(log_path, f"[INFO] Output directory: {output_directory}")
    update_log(log_path, f"[INFO] Embedding model: {embedding_model}")
    update_log(log_path, f"[INFO] Similarity threshold: {similarity_threshold}")
    update_log(log_path, f"[INFO] Linkage method: {linkage_method}")
    update_log(log_path, f"[INFO] LLM labeling enabled: {use_llm_labels} ({llm_model})")

    # ---------------------------------------------------------------------
    # 1. Load CSV
    # ---------------------------------------------------------------------
    df = pd.read_csv(input_file)
    update_log(log_path, f"[INFO] Loaded dataframe with {len(df)} rows.")

    # ---------------------------------------------------------------------
    # 2. Extract unique text items from requested columns
    # ---------------------------------------------------------------------
    items = extract_unique_items(df, column_names)
    update_log(log_path, f"[INFO] Extracted {len(items)} unique items from columns {column_names}.")

    if len(items) == 0:
        update_log(log_path, "[ERROR] No items found in selected columns.")
        raise ValueError("No items found to embed/group from the specified columns.")

    # ---------------------------------------------------------------------
    # 3. Generate / load embeddings (with caching and retry safety)
    # ---------------------------------------------------------------------
    update_log(log_path, "[INFO] Generating embeddings...")
    embeddings = embed_texts(
        texts=items,
        model=embedding_model,
        cache_path=cache_path,
        log_path=log_path,
        config=config
    )
    update_log(log_path, f"[INFO] Got embeddings with shape {embeddings.shape}.")

    # ---------------------------------------------------------------------
    # 4. Agglomerative Hierarchical Clustering
    # ---------------------------------------------------------------------
    update_log(log_path, "[INFO] Performing hierarchical clustering...")
    clusters, sim_df, Z = hierarchical_cluster_items(
        texts=items,
        embeddings=embeddings,
        similarity_threshold=similarity_threshold,
        linkage_method=linkage_method,
    )
    update_log(log_path, f"[INFO] Produced {len(clusters)} clusters via AHC.")

    # ---------------------------------------------------------------------
    # 5. Generate human-readable labels (optional)
    # ---------------------------------------------------------------------
    labels: Optional[Dict[int, str]] = None
    if use_llm_labels:
        update_log(log_path, "[INFO] Generating human-readable labels for clusters.")
        labels = label_clusters(
            clusters=clusters,
            model_name=llm_model,
            log_path=log_path,
            checkpoint_path=label_checkpoint_path,
            config=config,
        )
        update_log(log_path, f"[INFO] Labeled {len(labels)} clusters.")
    else:
        update_log(log_path, "[INFO] Skipping LLM labeling; using fallback Category_<id> labels.")

    # ---------------------------------------------------------------------
    # 6. Map cluster labels back to each row in the original dataframe
    # ---------------------------------------------------------------------
    update_log(log_path, "[INFO] Assigning category labels back to dataframe.")
    df_aug = assign_categories_to_dataframe(
        df=df,
        column_names=column_names,
        clusters=clusters,
        labels=labels,
    )

    update_log(
        log_path,
        "[INFO] Added category columns: "
        + ", ".join([f"{col}_category" for col in column_names])
    )

    # ---------------------------------------------------------------------
    # 7. Save augmented dataframe with new *_category columns
    # ---------------------------------------------------------------------
    augmented_path = output_directory / f"{save_name}_with_categories.csv"
    update_log(log_path, f"[INFO] Saving augmented dataset to {augmented_path}")
    df_aug.to_csv(augmented_path, index=False)

    # ---------------------------------------------------------------------
    # 8. Save category summary (cluster -> label, size, members)
    # ---------------------------------------------------------------------
    summary_records = []
    for cid, members in clusters.items():
        label_val = labels.get(cid, f"Category_{cid}") if labels else f"Category_{cid}"
        summary_records.append(
            {
                "Cluster_ID": cid,
                "Label": label_val,
                "Size": len(members),
                "Items": "; ".join(members),
            }
        )

    summary_df = pd.DataFrame(summary_records).sort_values(
        by=["Size"], ascending=False
    )

    mapping_path = output_directory / f"{save_name}_categories.csv"
    update_log(log_path, f"[INFO] Saving category summary to {mapping_path}")
    summary_df.to_csv(mapping_path, index=False)

    # ---------------------------------------------------------------------
    # 9. Save cosine similarity matrix between unique items
    # ---------------------------------------------------------------------
    sim_path = output_directory / f"{save_name}_similarity_matrix.csv"
    update_log(log_path, f"[INFO] Saving cosine similarity matrix to {sim_path}")
    sim_df.to_csv(sim_path, index=True)

    # ---------------------------------------------------------------------
    # 10. Save dendrogram plot of the AHC tree
    # ---------------------------------------------------------------------
    dendro_path = output_directory / f"{save_name}_dendrogram.png"
    update_log(log_path, f"[INFO] Saving dendrogram plot to {dendro_path}")
    save_dendrogram(
        linkage_matrix=Z,
        labels=items,
        out_path=dendro_path,
    )

    # ---------------------------------------------------------------------
    # 11. Wrap up
    # ---------------------------------------------------------------------
    update_log(log_path, "[INFO] === Grouping pipeline complete ===")
    update_log(log_path, f"[INFO] Rows in augmented output: {len(df_aug)}")
    update_log(log_path, f"[INFO] Clusters in summary: {len(summary_df)}")
    update_log(log_path, f"[INFO] Artifacts written:\n"
                         f" - {augmented_path}\n"
                         f" - {mapping_path}\n"
                         f" - {sim_path}\n"
                         f" - {dendro_path}\n")

    return df_aug
