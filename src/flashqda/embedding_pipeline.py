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

    Reads a CSV of extracted items and computes embeddings for specified text columns.
    Embeddings are saved to a JSON file for reuse. Supports incremental updates by 
    skipping already embedded items and logs progress to a project-aware location.

    Args:
        project (flashqda.ProjectContext): Project context providing result paths.
        config (flashqda.PipelineConfig, optional): Configuration containing extractable labels 
            (used as default `column_names` if none are provided).
        column_names (list of str, optional): Columns to embed. If not specified, defaults to 
            `config.extract_labels`.
        input_file (str or Path, optional): Path to CSV file with extracted items.
            Defaults to `project.results / "extracted.csv"`.
        output_directory (str or Path, optional): Directory to save the embeddings file and logs.
            Defaults to `project.results`.
        save_name (str, optional): Name of the JSON file to store embeddings.
            Defaults to `"embeddings.json"`.

    Returns:
        Path: Full path to the saved JSON file containing text embeddings.
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
        column_names = config.extract_labels  # e.g., ["cause", "effect"]

    embeddings = load_embeddings(output_file)

    for label in column_names:
        if label not in items.columns:
            continue
        for _, row in tqdm(items.iterrows(), total=len(items), desc=f"Embedding {label}s"):
            item = str(row[label]).strip()
            if pd.isna(row[label]):
                continue
            item = str(row[label]).strip()
            if not item or item in embeddings:
                continue

            emb = compute_embedding(item)
            embeddings[item] = emb
            save_embeddings(embeddings, output_file)
            update_log(log_file, f"Embedded {label}: {item}")

    save_embeddings(embeddings, output_file)
    num_docs = items["document_id"].nunique()
    print(f"Embedded items from {num_docs} documents.")
    return output_file