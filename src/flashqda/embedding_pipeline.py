import pandas as pd
from pathlib import Path

from flashqda.embeddings.provider import embed_texts
from flashqda.log_utils import update_log
from flashqda.pipelines.config import PipelineConfig


def embed_items(
        project,
        config: PipelineConfig = None,
        column_names=None,
        input_file=None,
        output_directory=None,
        save_name=None
        ):
    """
    Generate and cache embeddings for extracted text items (e.g., causes, effects).

    Collects all unique non-empty values from the specified columns, embeds them in
    batch using the configured provider, and persists them to a JSON cache file.
    Subsequent calls with the same cache path will only embed new, uncached texts.

    Args:
        project: ProjectContext providing default file paths.
        config: PipelineConfig with embedding_provider, embedding_model, and
            extract_fields (used as default column_names when not provided).
        column_names: List of column names whose values should be embedded.
            Defaults to config.extract_fields.
        input_file: Path to CSV containing the text columns. Defaults to
            project.results / "extracted.csv".
        output_directory: Directory for outputs and logs. Defaults to project.results.
        save_name: Filename for the embedding cache JSON.
            Defaults to "embeddings.json".

    Returns:
        Path to the embedding cache JSON file.
    """

    input_file = Path(input_file) if input_file else (project.results / "extracted.csv")
    output_directory = Path(output_directory) if output_directory else project.analysis_dir("embedding")
    save_name = save_name if save_name else "embeddings.json"
    output_file = output_directory / save_name
    output_directory.mkdir(parents=True, exist_ok=True)

    log_directory = output_directory / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    log_file = log_directory / f"{Path(save_name).stem}.log"

    if not column_names:
        column_names = config.extract_fields

    items = pd.read_csv(input_file)

    texts_to_embed = list({
        str(v).strip()
        for col in column_names
        if col in items.columns
        for v in items[col].dropna()
        if str(v).strip()
    })

    if not texts_to_embed:
        print("No items found to embed.")
        return output_file

    update_log(
        log_file,
        f"[INFO] Embedding {len(texts_to_embed)} unique items from columns {column_names}."
    )

    embed_texts(
        texts=texts_to_embed,
        cache_path=output_file,
        log_path=log_file,
        config=config,
    )

    num_docs = items["document_id"].nunique() if "document_id" in items.columns else "unknown"
    print(f"Embedded {len(texts_to_embed)} unique items from {num_docs} documents.")
    return output_file
