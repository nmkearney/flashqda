from pathlib import Path
import pandas as pd
from flashqda.prompt_loader import load_formatted_prompt
from flashqda.log_utils import update_log
from flashqda.pipelines.config import PipelineConfig
from flashqda.pipelines.validation import validate_columns
from flashqda.llm_utils import send_to_llm, extract_json_from_text
from tqdm.auto import tqdm
import json

def _pair_key(row_id, pair_id):
    return f"{int(row_id)}::{int(pair_id)}"

def _inject_ids_if_missing(items: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if "document_id" not in items.columns:
        print("Notice: 'document_id' column not found — assigning row-based IDs.")
        items.insert(0, "document_id", items.index + 1)
    id_col = f"{granularity}_id"
    if id_col not in items.columns:
        print(f"Notice: '{id_col}' column not found — assigning row-based IDs.")
        items.insert(1, id_col, items.index + 1)
    return items

DEFAULT_CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"}
    },
    "required": ["label"],
    "additionalProperties": False,
}

def _make_response_format(schema_name: str, schema: dict) -> dict:
    return{
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema,
        },
    }

def handle_classification(
        granularity, 
        item, 
        context_window, 
        prompt, 
        config
        ):
    schema = config.classify_schema or DEFAULT_CLASSIFY_SCHEMA

    filled_prompt = prompt.format(
        granularity=granularity,
        context_window="\n".join(context_window), 
        item=item,
        json_schema=json.dumps(schema, ensure_ascii=False)
        )
 
    response_text = send_to_llm(
        system_prompt=config.system_prompt,
        user_prompt=filled_prompt,
        config=config,
        response_format=_make_response_format("classification", schema),
    )

    try:
        response = extract_json_from_text(response_text)
        label = response.get("label")
        if label is None:
            print(f"[Warning] No 'label' found in response {response}")
            return "unknown"
        return label
    except Exception as e:
        print(f"Failed to parse response as JSON: {response_text}")
        return "unknown"

def get_label_name(label):
    if isinstance(label, str):
        return label.strip()
    elif isinstance(label, dict):
        return str(label.get("name", "")).strip()
    else:
        return str(label).strip()


def format_label_list_for_prompt(labels):
    lines = []

    for label in labels:
        if isinstance(label, str):
            name = label.strip()
            if name:
                lines.append(f"- {name}")

        elif isinstance(label, dict):
            name = str(label.get("name", "")).strip()
            description = str(label.get("description", "")).strip()

            if name and description:
                lines.append(f"- {name}: {description}")
            elif name:
                lines.append(f"- {name}")

        else:
            name = str(label).strip()
            if name:
                lines.append(f"- {name}")

    return "\n".join(lines)

def build_label_schema(labels, include_reasoning=False):
    allowed = sorted(set(
        name for name in (get_label_name(label) for label in labels) if name
    ))

    if "none" not in [l.lower() for l in allowed]:
        allowed.append("none")

    schema = {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": allowed
                }
            }
        },
        "required": ["labels"],
        "additionalProperties": False
    }

    if include_reasoning:
        schema["properties"]["reasoning"] = {
            "type": "object",
            "additionalProperties": {"type": "string"}
        }
        schema["required"].append("reasoning")

    return schema

def handle_labelling(granularity, 
                     item, 
                     context_window, 
                     prompt, 
                     label_list, 
                     config, 
                     pair=None, 
                     return_reasoning=False
                     ):
    active_labels = label_list if isinstance(label_list, list) else config.classify_labels
    schema = build_label_schema(active_labels, include_reasoning=return_reasoning)
    prompt_label_list = format_label_list_for_prompt(active_labels)

    filled_prompt = prompt.format(
        granularity=granularity,
        context_window="\n".join(context_window), 
        item=item, 
        label_list=prompt_label_list,
        pair=pair or "",
        return_reasoning=str(bool(return_reasoning)),
        json_schema=json.dumps(schema, ensure_ascii=False),
    )
    
    response_text = send_to_llm(
        system_prompt=config.system_prompt,
        user_prompt=filled_prompt,
        config=config,
        response_format=_make_response_format("labelling", schema),
    )

    try:
        response = extract_json_from_text(response_text)
        labels = response.get("labels", [])
        reasoning = response.get("reasoning", "")

        if isinstance(reasoning, dict):
            flat_reasoning = " | ".join(f"{k}: {v}" for k, v in reasoning.items())
        elif isinstance(reasoning, list):
            flat_reasoning = " ".join(str(x) for x in reasoning)
        else:
            flat_reasoning = str(reasoning or "")

        return (labels, flat_reasoning) if return_reasoning else labels

    except Exception as e:
        print(f"Failed to parse filter label response: {response_text}")
        if return_reasoning:
            return [], ""
        return []

def handle_extraction(
        granularity, 
        item, 
        context_window, 
        prompt, 
        config
        ):
    schema = config.extract_schema
 
    filled_prompt = prompt.format(
        granularity=granularity,
        context_window="\n".join(context_window), 
        item=item,
        json_schema=json.dumps(schema, ensure_ascii=False),
        )

    response_text = send_to_llm(
        system_prompt=config.system_prompt,
        user_prompt=filled_prompt,
        config=config,
        response_format=_make_response_format("extraction", schema),
    )
    try:
        response = extract_json_from_text(response_text)
        if not isinstance(response, dict):
            return {"relationships": []}
        
        relationships = response.get("relationships", [])
        if not isinstance(relationships, list):
            relationships = []
        
        normalized_relationships = []
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            
            normalized = {
                field: str(rel.get(field, "")).strip()
                for field in config.extract_fields
            }
            normalized_relationships.append(normalized)

        return {"relationships": normalized_relationships}
    
    except Exception as e:
        print(f"Failed to parse extraction response: {e}\nResponse: {response_text}")
        return {"relationships": []}

def parse_extractions_cell(val):
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    text = str(val).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []

def handle_refine_extraction(
        granularity,
        item,
        context_window,
        prior_relationships,
        prompt,
        config
        ):
    schema = config.extract_schema

    filled_prompt = prompt.format(
        granularity=granularity,
        context_window="\n".join(context_window),
        item=item,
        prior_relationships=json.dumps(prior_relationships, ensure_ascii=False),
        json_schema=json.dumps(schema, ensure_ascii=False),
    )

    response_text = send_to_llm(
        system_prompt=config.system_prompt,
        user_prompt=filled_prompt,
        config=config,
        response_format=_make_response_format("refine_extraction", schema),
    )

    try:
        response = extract_json_from_text(response_text)
        if not isinstance(response, dict):
            return {"relationships": []}

        relationships = response.get("relationships", [])
        if not isinstance(relationships, list):
            relationships = []

        normalized_relationships = []
        for rel in relationships:
            if not isinstance(rel, dict):
                continue

            normalized = {
                field: str(rel.get(field, "")).strip()
                for field in config.extract_fields
            }
            normalized_relationships.append(normalized)

        return {"relationships": normalized_relationships}

    except Exception:
        print(f"Failed to parse refinement response: {response_text}")
        return {"relationships": []}

def classify_items(
                     project=None, 
                     config: PipelineConfig = None, 
                     granularity=None,
                     context_length=1,
                     input_file=None, 
                     output_directory=None,
                     save_name=None
                     ):

    """
    Classify items (sentences or paragraphs) according to a set of criteria defined in a pipeline.

    Reads items from a CSV file, applies classification using a prompt-based pipeline, and writes
    the results to a new CSV file. Supports optional context windows and checkpointing for 
    resumability.

    Args:
        project (flashqda.ProjectContext): Project context for the file management and metadata.
        config (flashqda.PipelineConfig): The classification pipeline configuration. To use the 
            default pipeline, set `PipelineConfig.from_type = "causal"`. Custom pipelines are supported.
        granularity (str, optional): Segmentation level: "sentence" or "paragraph". Defaults to "sentence".
        context_length (int, optional): Number of prior items to include as context for classification.
            Defaults to 1.
        input_file (str, optional): Full path to the CSV file containing items to classify. The CSV should 
            have the columns: `document_id`, `filename`, `<granularity>_id`, `<granularity>`.
            If not provided, defaults to `project.results / "{granularity}s.csv"`.
        output_directory (str or Path, optional): Directory to save the results. Defaults to `project.results`.
        save_name (str, optional): Name of the output CSV file. Defaults to `"classified.csv"`.

    Returns:
        Path: The full path to the output CSV file containing the classification results.
    """

    granularity = granularity if granularity in ("sentence", "paragraph") else "sentence"
    input_file = input_file if input_file else (project.results / f"{granularity}s.csv")
    output_directory = Path(output_directory) if output_directory else project.analysis_dir("classification")
    save_name = save_name if save_name else f"classified.csv"
    output_file = output_directory / save_name
    output_directory.mkdir(parents=True, exist_ok=True)

    log_path = output_directory / "logs"
    log_path.mkdir(exist_ok=True)
    log_file = log_path / f"{Path(save_name)}.log"

    temp_path = output_directory / "temp"
    temp_path.mkdir(exist_ok=True)
    checkpoint_file = temp_path / f"{Path(save_name)}.checkpoint.json"

    items = pd.read_csv(input_file)
    items = _inject_ids_if_missing(items, granularity)
    validate_columns(items, ["document_id", f"{granularity}_id", f"{granularity}"], "classify_items")
    context_window = []

    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            processed = json.load(f)
    else:
        processed = {}

    prompt_file = config.prompt_files["classify"]
    prompt = load_formatted_prompt(prompt_file, project=project, config=config)

    write_header = not output_file.exists() or output_file.stat().st_size == 0

    updated_count = 0

    for idx, row in tqdm(items.iterrows(), total=len(items), desc="Classifying"):
        doc_id = str(row.get("document_id", "unknown"))
        filename = str(row.get("filename", "unknown"))
        row_id = int(row.get(f"{granularity}_id", idx + 1))

        if doc_id in processed and row_id in processed[doc_id]:
            continue

        # Rebuild context window from previous N items in same document
        start = max(idx - context_length, 0)
        context_window = [
            items.iloc[j][f"{granularity}"]
            for j in range(start, idx)
            if str(items.iloc[j].get("document_id", "unknown")) == doc_id
        ]

        label = handle_classification(
            granularity=granularity,
            item=row[f"{granularity}"],
            context_window=context_window, 
            prompt=prompt, 
            config=config
            )
        row_result = {
            "document_id": doc_id,
            "filename": filename,
            f"{granularity}_id": row_id,
            f"{granularity}": row[f"{granularity}"],
            "classification": label
        }
        pd.DataFrame([row_result]).to_csv(output_file, mode='a', index=False, header=write_header)
        write_header = False

        processed.setdefault(doc_id, []).append(row_id)

        updated_count += 1

        with open(checkpoint_file, "w") as f:
            json.dump(processed, f)

        update_log(log_file, f"Classified {granularity} {row_id} in document {doc_id} as {label}")

    num_docs = items["document_id"].nunique()
    print(f"Classified {updated_count} items in {num_docs} documents.")
    return output_file

def label_items(
        project=None, 
        config=None,
        granularity=None,
        context_length=1,
        label_column="classification",
        include_class=None, 
        label_list=None,
        on_classified = False,
        on_extracted = False,
        expand=False,
        capture_reasoning=False,
        input_file=None, 
        output_directory=None,
        save_name=None
        ):

    """
        Label items (sentences, paragraphs, or abstracts) with one or more filter tags.

        Reads a CSV file, applies a prompt-based labeling step using the specified pipeline, 
        and writes updated labels to a new CSV file. Supports contextual labeling, checkpointing, 
        and optional label column expansion for easier postprocessing.

        Args:
            project (flashqda.ProjectContext): Project context for file management and metadata.
            config (flashqda.PipelineConfig): Pipeline configuration, including prompt files and valid labels.
            granularity (str, optional): Unit of labeling. Options: "sentence", "paragraph", or "abstract".
                Defaults to "sentence".
            context_length (int, optional): Number of previous items to include as context. Defaults to 1.
            label_column (str, optional): Column containing classification labels.
            include_class (str, optional): Only items with this classification label will be considered for labeling.
                Defaults to the first label in `config.classify_labels`.
            label_list (list of str, optional): List of labels to apply to items.
            on_classified (bool, optional): If True, applies labelling to classified item pairs.
            on_extracted (bool, optional): If True, applies labeling to extracted item pairs using 
                pair metadata (i.e., original sentence or paragraph). Defaults to False.
            expand (bool, optional): If True, adds one-hot encoded columns for each unique label.
                Defaults to False.
            capture_reasoning (bool, optional): If True, captures the reasoning used by the LLM to assign the labels.
            input_file (str or Path, optional): Path to input CSV. Defaults to 
                `project.results / "classified.csv"`.
            output_directory (str or Path, optional): Directory where output is saved. Defaults to `project.results`.
            save_name (str, optional): Name of the labeled output CSV file. Defaults to 
                `"labelled.csv"`.

        Returns:
            Path: Path to the labeled output CSV file with new or updated filter tags.
        """

    granularity = granularity if granularity in ("sentence", "paragraph") else "sentence"
    input_file = input_file if input_file else (project.results / "classified.csv")
    output_directory = Path(output_directory) if output_directory else project.analysis_dir("labelling")
    save_name = save_name if save_name else "labelled.csv"
    output_file = output_directory / save_name
    output_directory.mkdir(parents=True, exist_ok=True)
    include_class = include_class if include_class else get_label_name(config.classify_labels[0])

    items = pd.read_csv(input_file)
    items.columns = [col.lower() for col in items.columns]
    label_column = (label_column or "classification").lower()

    items = _inject_ids_if_missing(items, granularity)
    validate_columns(items, ["document_id", f"{granularity}_id", f"{granularity}"], "label_items")
    items[f"{granularity}"] = items[f"{granularity}"].astype(str)

    filter_col = f"filter_labels_{Path(save_name).stem}"
    if filter_col not in items.columns:
        items[filter_col] = ""

    # Add reasoning column only if requested
    if capture_reasoning:
        reasoning_col_name = f"labels_reasoning_{Path(save_name).stem}"
        if reasoning_col_name not in items.columns:
            items[reasoning_col_name] = ""

    # Setup paths
    log_path = output_directory / "logs"
    log_path.mkdir(exist_ok=True)
    log_file = log_path / f"{Path(save_name).stem}.log"

    temp_path = output_directory / "temp"
    temp_path.mkdir(exist_ok=True)
    checkpoint_file = temp_path / f"{Path(save_name).stem}.checkpoint.json"

    context_window = []

    # Load checkpoint
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            processed = json.load(f)
    else:
        processed = {}

    if on_extracted:
        key = "label_extracted"
    else:
        key = "label_sent_para"
 
    prompt_file = config.prompt_files[key]
    prompt = load_formatted_prompt(prompt_file, project=project, config=config)

    updated_count = 0

    for i, row in tqdm(items.iterrows(), total=len(items), desc="Filter Labeling"):
        doc_id = str(row.get("document_id", "unknown"))
        row_id = int(row.get(f"{granularity}_id", -1))

        if on_extracted:
            pair_val = row.get("pair_id")
            pair_id = int(pair_val) if pd.notnull(pair_val) else 0
        else:
            pair_id = None

        if on_extracted:
            pair_key = _pair_key(row_id, pair_id)
            if doc_id in processed and pair_key in processed[doc_id]:
                continue
        else:
            if doc_id in processed and row_id in processed[doc_id]:
                continue
        
        if on_classified:
            classification = str(row[f"{label_column}"]).strip().lower()
            target = str(include_class).strip().lower()
            if classification != target:
                continue

        # Rebuild context window from previous N items in same document
        start = max(i - context_length, 0)
        context_window = [
            items.iloc[j][f"{granularity}"]
            for j in range(start, i)
            if str(items.iloc[j].get("document_id", "unknown")) == doc_id
        ]

        # Backfill logic: if labels exists but we're newly capturing reasoning, allow pass-through
        labels_exist = isinstance(row[filter_col], str) and row[filter_col].strip()
        need_reasoning_backfill = capture_reasoning and (not str(row.get(reasoning_col_name, "")).strip())

        if labels_exist and not need_reasoning_backfill:
            continue

        if on_extracted:

            first_part, second_part = config.extract_fields  # e.g. ["gain", "cost"]

            first_val = str(row.get(first_part) or "").strip().lower()
            second_val = str(row.get(second_part) or "").strip().lower()

            if not first_val or first_val == "nan":  # require first_part at least
                continue

            pair_text = f"{first_part.capitalize()}: {first_val}\n{second_part.capitalize()}: {second_val}"

        else:
            pair_text = ""

        # Call model once
        if capture_reasoning:
            labels, reasoning = handle_labelling(
                granularity=granularity,
                item=row[f"{granularity}"],
                context_window=context_window,
                prompt=prompt,
                label_list=label_list,
                config=config,
                pair=pair_text,
                return_reasoning=True
            )
        else:
            labels = handle_labelling(
                granularity=granularity,
                item=row[f"{granularity}"],
                context_window=context_window,
                prompt=prompt,
                label_list=label_list,
                config=config,
                pair=pair_text
            )
            reasoning = ""

        # If we're only backfilling reasoning, keep existing labels
        if labels_exist and need_reasoning_backfill:
            items.at[i, reasoning_col_name] = str(reasoning or "").strip()
        else:
            raw_labels = [label.strip() for label in labels if label.strip()]
            items.at[i, filter_col] = ", ".join(raw_labels)
            if capture_reasoning:
                items.at[i, reasoning_col_name] = str(reasoning or "").strip()

        items.to_csv(output_file, index=False)

        if on_extracted:
            processed.setdefault(doc_id, []).append(_pair_key(row_id, pair_id))
        else:
            processed.setdefault(doc_id, []).append(row_id)

        with open(checkpoint_file, "w") as f:
            json.dump(processed, f)
        updated_count += 1

        log_msg = (
            f"Labeled filters for pair {pair_id} in {granularity} {row_id} (doc {doc_id}): {labels}"
            if on_extracted else
            f"Labeled filters for {granularity} {row_id} in document {doc_id}: {labels}"
        )
        if capture_reasoning and reasoning:
            log_msg += f" | reasoning: {reasoning}"
        update_log(log_file, log_msg)

    if expand:
        # Extract all labels used (excluding 'none')
        all_labels = set()
        for tags in items[filter_col].dropna():
            all_labels.update(
                label.strip() for label in tags.split(",")
                if label.strip() and label.strip().lower() != "none"
            )

        for label in sorted(all_labels):
            if not label.strip():
                continue
            col_name = f"{label.strip().lower().replace(' ', '_')}"
            items[col_name] = items[filter_col].apply(
                lambda val: label.lower() in [x.strip().lower() for x in val.split(",")] if isinstance(val, str) else False
            )

        # Final write with expanded labels
        items.to_csv(output_file, index=False)

    num_docs = items["document_id"].nunique()
    print(f"Labelled {updated_count} items in {num_docs} documents.")
    return output_file

def extract_from_classified(
        project=None, 
        config: PipelineConfig = None, 
        granularity=None,
        context_length=1,
        label_column=None, 
        include_class=None,
        filter_keys=None, 
        filter_column=None, 
        input_file=None, 
        output_directory=None,
        save_name=None
        ):

    """
    Extract structured relationships (e.g., cause-effect pairs) from classified items.

    Reads a CSV of pre-classified text segments (sentences or paragraphs), filters them
    based on classification and optional label criteria, and uses a prompt-based pipeline to 
    extract relationships. Supports context windows, checkpointing, and progressive saving 
    of results to a CSV file.

    Args:
        project (flashqda.ProjectContext): Project context providing paths and metadata.
        config (flashqda.PipelineConfig): Configuration for prompts and extractable fields.
        granularity (str, optional): Unit of analysis, "sentence" or "paragraph". Defaults to "sentence".
        context_length (int, optional): Number of prior items to include as context. Defaults to 1.
        label_column (str, optional): Column containing classification labels.
        include_class (str, optional): Classification label required for an item to be eligible for extraction.
            Defaults to the first label in `config.classify_labels`.
        filter_keys (str or list, optional): Labels to exclude from extraction (e.g., items labeled as "none").
            Items with these labels in `filter_column` are skipped.
        filter_column (str, optional): Name of the column containing filter keys (e.g., "filter_labels").
        input_file (str or Path, optional): Path to the classified input CSV.
            Defaults to `project.results / "classified.csv"`.
        output_directory (str or Path, optional): Directory where the output CSV will be saved.
            Defaults to `project.results`.
        save_name (str, optional): Filename for the output CSV file.
            Defaults to `"extracted.csv"`.

    Returns:
        Path: The full path to the CSV file containing extracted relationships for each qualifying item.
    """

    granularity = granularity if granularity in ("sentence", "paragraph") else "sentence"
    input_file = input_file if input_file else project.results / "classified.csv"
    output_directory = Path(output_directory) if output_directory else project.analysis_dir("extraction")
    save_name = save_name if save_name else "extracted.csv"
    output_file = output_directory / save_name
    output_directory.mkdir(parents=True, exist_ok=True)
    include_class = include_class if include_class else get_label_name(config.classify_labels[0])

    log_path = output_directory / "logs"
    log_path.mkdir(exist_ok=True)
    log_file = log_path / f"{Path(save_name).stem}.log"

    temp_path = output_directory / "temp"
    temp_path.mkdir(exist_ok=True)
    checkpoint_file = temp_path / f"{Path(save_name).stem}.checkpoint.json"

    items = pd.read_csv(input_file)
    items.columns = [col.lower() for col in items.columns]
    items = _inject_ids_if_missing(items, granularity)
    validate_columns(items, ["document_id", f"{granularity}_id", f"{granularity}"], "extract_from_classified")
    label_column = (label_column or "classification").lower()
    filter_column = filter_column.lower() if filter_column else None
    context_window = []

    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            processed = json.load(f)
    else:
        processed = {}

    prompt_file = config.prompt_files["extract"]
    prompt = load_formatted_prompt(prompt_file, project=project, config=config)

    write_header = not output_file.exists() or output_file.stat().st_size == 0

    updated_count = 0

    for idx, row in tqdm(items.iterrows(), total=len(items), desc="Extracting"):
        doc_id = str(row.get("document_id", "unknown"))
        filename = str(row.get("filename", "unknown"))
        row_id = int(row.get(f"{granularity}_id", -1))

        # Skip if already processed
        if doc_id in processed and row_id in processed[doc_id]:
            continue

        # Rebuild context window from previous N items in same document
        start = max(idx - context_length, 0)
        context_window = [
            items.iloc[j][f"{granularity}"]
            for j in range(start, idx)
            if str(items.iloc[j].get("document_id", "unknown")) == doc_id
        ]

        # Start with all columns from the original row
        result_row = row.to_dict()

        # Normalize ID fields
        result_row["document_id"] = doc_id
        result_row["filename"] = filename
        result_row[f"{granularity}_id"] = row_id

        classification = str(row[f"{label_column}"]).strip().lower()
        target = str(include_class).strip().lower()

        if classification == target: # Include items of the chosen type (e.g., "causal")
            filter_val = str(row.get(filter_column, "")).strip().lower()
            should_extract = False
            if not filter_keys:
                should_extract=True
            elif isinstance(filter_keys, str):
                should_extract = filter_val != filter_keys.lower()
            elif isinstance(filter_keys, (list, set, tuple)):
                should_extract = filter_val not in [str(x).lower() for x in filter_keys]
            if should_extract:
                # Proceed with extraction
                response = handle_extraction(
                    granularity=granularity,
                    item=row[f"{granularity}"], 
                    context_window=context_window, 
                    prompt=prompt, 
                    config=config
                    )
                relationships = response.get("relationships", [])
                result_row["extractions_json"] = json.dumps(relationships, ensure_ascii=False)
                result_row["n_extractions"] = len(relationships)
                pd.DataFrame([result_row]).to_csv(
                    output_file, mode='a', index=False, header=write_header
                    )
                write_header = False

                updated_count += 1
                update_log(
                    log_file, 
                    f"Processed {granularity} {row_id} in document {doc_id}: {relationships}"
                    )
            else:
                # Append causal row with filters with empty extractions
                result_row["extractions_json"] = json.dumps([], ensure_ascii=False)
                result_row["n_extractions"] = 0
                pd.DataFrame([result_row]).to_csv(
                    output_file, mode='a', index=False, header=write_header
                    )
                write_header = False
        else:
            # Append non-causal row with empty extractions
            result_row["extractions_json"] = json.dumps([], ensure_ascii=False)
            result_row["n_extractions"] = 0
            pd.DataFrame([result_row]).to_csv(
                output_file, mode='a', index=False, header=write_header
                )
            write_header = False

        processed.setdefault(doc_id, []).append(row_id)

        with open(checkpoint_file, "w") as f:
            json.dump(processed, f)

    num_docs = items["document_id"].nunique()
    print(f"Extracted from {updated_count} items in {num_docs} documents.")
    return output_file

def refine_extracted(
        project=None,
        config: PipelineConfig = None,
        granularity=None,
        context_length=1,
        input_file=None,
        extraction_column="extractions_json",
        output_directory=None,
        save_name=None,
        skip_empty=True,
        overwrite=False
        ):

    """
    Validate and correct previously extracted relationships using the LLM.

    Reads a CSV produced by ``extract_from_classified``, passes each item's
    existing extractions back through the LLM for review, and writes a new
    CSV with a ``refined_extractions_json`` column containing the corrected
    output. Supports context windows, checkpointing, and skipping items
    with no prior extractions.

    Args:
        project (flashqda.ProjectContext): Project context providing paths and metadata.
        config (flashqda.PipelineConfig): Configuration containing the ``refine_extract``
            prompt file key and LLM settings.
        granularity (str, optional): Unit of analysis — ``"sentence"`` or ``"paragraph"``.
            Defaults to ``"sentence"``.
        context_length (int, optional): Number of prior items to include as context.
            Defaults to ``1``.
        input_file (str or Path, optional): Path to the CSV produced by
            ``extract_from_classified``. Defaults to ``project.results / "extracted.csv"``.
        extraction_column (str, optional): Column name in ``input_file`` that holds the
            raw JSON extractions. Defaults to ``"extractions_json"``.
        output_directory (str or Path, optional): Directory where the refined CSV will be
            saved. Defaults to a fresh timestamped ``analyses/extraction/`` directory.
        save_name (str, optional): Filename for the output CSV. Defaults to
            ``"refined_extracted.csv"``.
        skip_empty (bool, optional): If ``True``, items with no existing extractions are
            written as empty lists without calling the LLM. Defaults to ``True``.
        overwrite (bool, optional): If ``True``, re-process items that already have a
            refined result. Defaults to ``False``.

    Returns:
        Path: Full path to the output CSV file containing refined extractions.
    """

    granularity = granularity if granularity in ("sentence", "paragraph") else "sentence"
    input_file = input_file if input_file else project.results / "extracted.csv"
    output_directory = Path(output_directory) if output_directory else project.analysis_dir("extraction")
    save_name = save_name if save_name else "refined_extracted.csv"
    output_file = output_directory / save_name
    output_directory.mkdir(parents=True, exist_ok=True)

    log_path = output_directory / "logs"
    log_path.mkdir(exist_ok=True)
    log_file = log_path / f"{Path(save_name).stem}.log"

    temp_path = output_directory / "temp"
    temp_path.mkdir(exist_ok=True)
    checkpoint_file = temp_path / f"{Path(save_name).stem}.checkpoint.json"

    items = pd.read_csv(input_file)
    items.columns = [col.lower() for col in items.columns]
    items = _inject_ids_if_missing(items, granularity)
    validate_columns(items, ["document_id", f"{granularity}_id", f"{granularity}"], "refine_extracted")
    extraction_column = extraction_column.lower()

    refined_col = "refined_extractions_json"
    refined_n_col = "n_refined_extractions"

    if refined_col not in items.columns:
        items[refined_col] = ""
    if refined_n_col not in items.columns:
        items[refined_n_col] = 0

    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            processed = json.load(f)
    else:
        processed = {}

    prompt_file = config.prompt_files["refine_extract"]
    prompt = load_formatted_prompt(prompt_file, project=project, config=config)

    updated_count = 0

    for idx, row in tqdm(items.iterrows(), total=len(items), desc="Refining extractions"):
        doc_id = str(row.get("document_id", "unknown"))
        row_id = int(row.get(f"{granularity}_id", -1))

        if doc_id in processed and row_id in processed[doc_id] and not overwrite:
            continue

        existing_refined = str(row.get(refined_col, "")).strip()
        if existing_refined and not overwrite:
            continue

        prior_relationships = parse_extractions_cell(row.get(extraction_column, ""))

        if skip_empty and not prior_relationships:
            items.at[idx, refined_col] = json.dumps([], ensure_ascii=False)
            items.at[idx, refined_n_col] = 0
            processed.setdefault(doc_id, []).append(row_id)
            with open(checkpoint_file, "w") as f:
                json.dump(processed, f)
            continue

        start = max(idx - context_length, 0)
        context_window = [
            items.iloc[j][f"{granularity}"]
            for j in range(start, idx)
            if str(items.iloc[j].get("document_id", "unknown")) == doc_id
        ]

        response = handle_refine_extraction(
            granularity=granularity,
            item=row[f"{granularity}"],
            context_window=context_window,
            prior_relationships=prior_relationships,
            prompt=prompt,
            config=config
        )

        refined_relationships = response.get("relationships", [])
        items.at[idx, refined_col] = json.dumps(refined_relationships, ensure_ascii=False)
        items.at[idx, refined_n_col] = len(refined_relationships)

        items.to_csv(output_file, index=False)

        processed.setdefault(doc_id, []).append(row_id)
        with open(checkpoint_file, "w") as f:
            json.dump(processed, f)

        updated_count += 1
        update_log(
            log_file,
            f"Refined {granularity} {row_id} in document {doc_id}: {refined_relationships}"
        )

    if not output_file.exists():
        items.to_csv(output_file, index=False)

    num_docs = items["document_id"].nunique()
    print(f"Refined extractions for {updated_count} items in {num_docs} documents.")
    return output_file

def parse_extractions_cell(val):
    if pd.isna(val):
        return []

    if isinstance(val, list):
        return val

    text = str(val).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []
    
def explode_extractions(
        project=None,
        config: PipelineConfig = None,
        granularity=None,
        input_file=None,
        extraction_column=None,
        output_directory=None,
        save_name=None,
        keep_empty_rows=True
        ):
    """
    Explode list-valued extraction output into one row per extracted item.

    Reads a CSV where each source row contains a JSON list of extracted items
    and writes a new CSV with one row per extracted item.

    Args:
        project: flashQDA project context.
        config: PipelineConfig containing extract_fields.
        granularity: "sentence" or "paragraph".
        input_file: Source-level extraction CSV.
        extraction_column: Column containing JSON list. Defaults to
            "refined_extractions_json" if present, otherwise "extractions_json".
        output_directory: Directory for output CSV.
        save_name: Output filename. Defaults to "extracted_exploded.csv".
        keep_empty_rows: If True, keep rows with no extractions as blank extraction rows.

    Returns:
        Path to exploded CSV.
    """

    granularity = granularity if granularity in ("sentence", "paragraph") else "sentence"
    input_file = input_file if input_file else project.results / "refined_extracted.csv"
    output_directory = Path(output_directory) if output_directory else project.analysis_dir("extraction")
    save_name = save_name if save_name else "extracted_exploded.csv"
    output_file = output_directory / save_name
    output_directory.mkdir(parents=True, exist_ok=True)

    items = pd.read_csv(input_file)
    items.columns = [col.lower() for col in items.columns]

    # Choose extraction column automatically if not supplied
    if extraction_column is None:
        if "refined_extractions_json" in items.columns:
            extraction_column = "refined_extractions_json"
        elif "extractions_json" in items.columns:
            extraction_column = "extractions_json"
        else:
            raise ValueError(
                "No extraction column found. Expected 'refined_extractions_json' "
                "or 'extractions_json'."
            )
    else:
        extraction_column = extraction_column.lower()

    if extraction_column not in items.columns:
        raise ValueError(f"Extraction column '{extraction_column}' not found.")

    extract_fields = list(config.extract_fields)

    exploded_rows = []

    for idx, row in items.iterrows():
        base_row = row.to_dict()
        extractions = parse_extractions_cell(row.get(extraction_column, ""))

        # Remove list-valued columns from exploded output? Optional.
        # I would keep them for provenance for now.
        if extractions:
            for pair_id, extraction in enumerate(extractions, start=1):
                exploded_row = base_row.copy()
                exploded_row["pair_id"] = pair_id

                for field in extract_fields:
                    if isinstance(extraction, dict):
                        exploded_row[field] = str(extraction.get(field, "")).strip()
                    else:
                        exploded_row[field] = ""

                exploded_rows.append(exploded_row)

        elif keep_empty_rows:
            exploded_row = base_row.copy()
            exploded_row["pair_id"] = ""

            for field in extract_fields:
                exploded_row[field] = ""

            exploded_rows.append(exploded_row)

    exploded_df = pd.DataFrame(exploded_rows)
    exploded_df.to_csv(output_file, index=False)

    print(
        f"Exploded {len(exploded_rows)} extracted rows "
        f"from {len(items)} source rows."
    )

    return output_file