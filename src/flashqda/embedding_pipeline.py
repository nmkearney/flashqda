import pandas as pd
from flashqda.embedding_core import compute_embedding
from flashqda.embedding_cache import load_embeddings, save_embeddings
from flashqda.pipelines.config import PipelineConfig
from flashqda.log_utils import update_log
from pathlib import Path
from tqdm import tqdm


def embed_items(
        project, 
        config: PipelineConfig = None, 
        column_names=None, 
        input_file=None, 
        output_directory=None,
        save_name=None
        ):
    """
    Generate and save embeddings for extracted text items (e.g., causes, effects).
    Supports resume after interruption and avoids recomputation of existing items.
    """

    input_file = Path(input_file) if input_file else (project.results / "extracted.csv")
    output_directory = Path(output_directory) if output_directory else project.results
    save_name = save_name if save_name else "embeddings.json"
    output_file = output_directory / save_name
    output_directory.mkdir(parents=True, exist_ok=True)

    log_directory = output_directory / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    log_file = log_directory / f"{Path(save_name).stem}.log"

    items = pd.read_csv(input_file)
    if not column_names:
        column_names = config.extract_labels

    embeddings = load_embeddings(output_file)

    # Collect current valid items from the CSV
    valid_items = set()
    for label in column_names:
        if label in items.columns:
            valid_items.update(
                str(row[label]).strip()
                for _, row in items.iterrows()
                if pd.notna(row[label]) and str(row[label]).strip()
            )

    # Prune outdated embeddings
    original_keys = set(embeddings.keys())
    removed = original_keys - valid_items
    if removed:
        for key in removed:
            del embeddings[key]
        update_log(log_file, f"Removed {len(removed)} outdated embeddings.")

    for label in column_names:
        if label not in items.columns:
            continue

        new_embeddings = {}
        for _, row in tqdm(items.iterrows(), total=len(items), desc=f"Embedding {label}s"):
            if pd.isna(row[label]):
                continue
            item = str(row[label]).strip()
            if not item or item in embeddings:
                continue

            emb = compute_embedding(item)
            new_embeddings[item] = emb
            update_log(log_file, f"Embedded {label}: {item}")

        # Update and save after each label
        embeddings.update(new_embeddings)
        save_embeddings(embeddings, output_file)

    num_docs = items["document_id"].nunique() if "document_id" in items.columns else "unknown"
    print(f"Embedded items from {num_docs} documents.")
    return output_file
