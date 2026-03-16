import pandas as pd
from flashqda.pipelines.config import PipelineConfig
from flashqda.embedding_cache import load_embeddings
from flashqda.log_utils import update_log
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import numpy as np

def link_items(
        project,
        config,
        threshold=0.85,
        input_file=None,
        embedding_file=None,
        output_directory=None,
        save_name=None
        ):
    """
    Link effects to potential causes based on semantic similarity between text embeddings.

    Compares embeddings of extracted cause-effect items to identify likely causal links 
    using cosine similarity. Links with similarity above a specified threshold are recorded 
    in a structured CSV file, with associated metadata for traceability.

    Args:
        project (flashqda.ProjectContext): Project context providing file paths.
        config (flashqda.PipelineConfig): Pipeline configuration with extract labels (e.g., ["cause", "effect"]).
        threshold (float, optional): Cosine similarity threshold for linking items.
            Only pairs above this value are retained. Defaults to 0.85.
        input_file (str or Path, optional): Path to the CSV file containing extracted cause/effect items.
            Defaults to `project.results / "extracted.csv"`.
        embedding_file (str or Path, optional): Path to the JSON file containing item embeddings.
            Defaults to `project.results / "embeddings.json"`.
        output_directory (str or Path, optional): Directory to save the linked results CSV and logs.
            Defaults to `project.results`.
        save_name (str, optional): Filename for the output CSV file containing suggested links.
            Defaults to `"suggested_links.csv"`.

    Returns:
        Path: Full path to the CSV file containing effect-cause link suggestions based on similarity.
    """
    input_file = Path(input_file) if input_file else (project.results / "extracted.csv")
    embedding_file = Path(embedding_file) if embedding_file else (project.results / "embeddings.json")
    output_directory = Path(output_directory) if output_directory else project.results
    save_name = save_name if save_name else "suggested_links.csv"
    output_file = output_directory / save_name
    output_directory.mkdir(parents=True, exist_ok=True)

    log_directory = output_directory / "logs"
    log_directory.mkdir(exist_ok=True)
    log_file = log_directory / f"{Path(save_name).stem}.log"

    items = pd.read_csv(input_file)
    embeddings = load_embeddings(embedding_file)

    cause_label, effect_label = config.extract_labels
    items = items.dropna(subset=[cause_label, effect_label])

    # Get unique causes/effects with available embeddings
    causes = items[cause_label].unique().tolist()
    effects = items[effect_label].unique().tolist()

    cause_texts = [c for c in causes if c in embeddings]
    effect_texts = [e for e in effects if e in embeddings]

    cause_vectors = np.array([embeddings[c] for c in cause_texts])
    effect_vectors = np.array([embeddings[e] for e in effect_texts])

    similarities = cosine_similarity(effect_vectors, cause_vectors)

    # Build metadata index
    effect_meta = items.groupby(effect_label).first().to_dict(orient="index")
    cause_meta = items.groupby(cause_label).first().to_dict(orient="index")

    rows = []
    for i, effect in enumerate(effect_texts):
        sim_scores = similarities[i]
        for j, score in enumerate(sim_scores):
            if score < threshold:
                continue

            cause = cause_texts[j]
            from_meta = effect_meta.get(effect, {})
            to_meta = cause_meta.get(cause, {})

            if effect == cause:
                continue  # skip tautologies

            # Optionally, skip same-sentence links too:
            if from_meta.get("sentence_id") == to_meta.get("sentence_id") and \
            from_meta.get("document_id") == to_meta.get("document_id"):
                continue

            rows.append({
                "from_effect_sentence": from_meta.get("sentence", ""),
                "from_cause": from_meta.get("cause", ""),
                "from_effect": effect,
                "to_cause": cause,
                "to_effect": to_meta.get("effect", ""),
                "to_cause_sentence": to_meta.get("sentence", ""),
                "similarity": score,
                "from_effect_document_id": from_meta.get("document_id", ""),
                "from_effect_filename": from_meta.get("filename", ""),
                "from_effect_sentence_id": from_meta.get("sentence_id", ""),
                "to_cause_document_id": to_meta.get("document_id", ""),
                "to_cause_filename": to_meta.get("filename", ""),
                "to_cause_sentence_id": to_meta.get("sentence_id", "")
            })

            update_log(log_file, f"Linked '{effect}' -> '{cause}' ({score:.2f})")

    pd.DataFrame(rows).to_csv(output_file, index=False)
    num_docs = items["document_id"].nunique()
    print(f"Linked items by semantic similarity in {num_docs} documents.")
    return output_file
