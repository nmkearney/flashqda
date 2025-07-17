from pathlib import Path
import pandas as pd
from flashqda.prompt_loader import load_formatted_prompt
from flashqda.log_utils import update_log
from flashqda.pipelines.config import PipelineConfig
from flashqda.openai_utils import send_to_openai
from tqdm.notebook import tqdm
import json

def handle_classification(granularity, item, context_window, prompt, config):
    filled_prompt = prompt.format(granularity=granularity,
                                  context_window="\n".join(context_window), 
                                  item=item
                                  )
    response_text = send_to_openai(
        system_prompt=config.system_prompt,
        user_prompt=filled_prompt,
        response_format={"type": "json_object"}
    )

    try:
        response = json.loads(response_text)
        label = response.get("label")
        if label is None:
            print(f"[Warning] No 'label' found in response {response}")
            return "unknown"
        return label
    except Exception as e:
        print(f"Failed to parse response as JSON: {response_text}")
        return "unknown"

def handle_labelling(granularity, item, context_window, prompt, label_list, config, pair=None):
    filled_prompt = prompt.format(
        granularity=granularity,
        context_window="\n".join(context_window), 
        item=item, 
        label_list=label_list,
        pair=pair or "")
    response_text = send_to_openai(
        system_prompt=config.system_prompt,
        user_prompt=filled_prompt,
        response_format={"type": "json_object"}
    )

    try:
        response = json.loads(response_text)
        return response.get("labels", [])
    except Exception as e:
        print(f"Failed to parse filter label response: {response_text}")
        return []

def handle_extraction(granularity, item, context_window, prompt, config):
    filled_prompt = prompt.format(
        granularity=granularity,
        context_window="\n".join(context_window), 
        item=item
        )
    response_text = send_to_openai(
        system_prompt=config.system_prompt,
        user_prompt=filled_prompt,
        response_format={"type": "json_object"}
    )
    try:
        return json.loads(response_text)
    except Exception as e:
        print(f"Failed to parse extraction response: {response_text}")
        return {"relationships": [{}]}

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
    output_directory = Path(output_directory) if output_directory else project.results
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
    context_window = []

    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            processed = json.load(f)
    else:
        processed = {}

    prompt_file = config.prompt_files["classify"]
    prompt = load_formatted_prompt(prompt_file, project=project)

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
        include_class=None, 
        label_list=None,
        on_extracted = False,
        expand=False,
        input_file=None, 
        output_directory=None,
        save_name=None
        ):

    """
        Label classified items (sentences, paragraphs, or abstracts) with one or more filter tags.

        Reads a classified CSV file, applies a prompt-based labeling step using the specified pipeline, 
        and writes updated labels to a new CSV file. Supports contextual labeling, checkpointing, 
        and optional label column expansion for easier postprocessing.

        Args:
            project (flashqda.ProjectContext): Project context for file management and metadata.
            config (flashqda.PipelineConfig): Pipeline configuration, including prompt files and valid labels.
            granularity (str, optional): Unit of labeling. Options: "sentence", "paragraph", or "abstract".
                Defaults to "sentence".
            context_length (int, optional): Number of previous items to include as context. Defaults to 1.
            include_class (str, optional): Only items with this classification label will be considered for labeling.
                Defaults to the first label in `config.labels`.
            label_list (list of str, optional): List of labels to apply to items.
            on_extracted (bool, optional): If True, applies labeling to extracted item pairs using 
                pair metadata (i.e., original sentence or paragraph). Defaults to False.
            expand (bool, optional): If True, adds one-hot encoded columns for each unique label.
                Defaults to False.
            input_file (str or Path, optional): Path to input CSV. Defaults to 
                `project.results / "classified.csv"`.
            output_directory (str or Path, optional): Directory where output is saved. Defaults to `project.results`.
            save_name (str, optional): Name of the labeled output CSV file. Defaults to 
                `"labelled.csv"`.

        Returns:
            Path: Path to the labeled output CSV file with new or updated filter tags.
        """

    granularity = granularity if granularity in ("sentence", "paragraph", "abstract") else "sentence"
    input_file = input_file if input_file else (project.results / "classified.csv")
    output_directory = Path(output_directory) if output_directory else project.results
    save_name = save_name if save_name else (project.results / "labelled.csv")
    output_file = output_directory / save_name
    output_directory.mkdir(parents=True, exist_ok=True)
    include_class = include_class if include_class else config.labels[0]

    items = pd.read_csv(input_file)
    items.columns = [col.lower() for col in items.columns]

    if granularity == "abstract":
        items["document_id"] = items.index + 1
        items["abstract_id"] = items.index + 1
        items["abstract"] = items["abstract"].astype(str)

    filter_col = f"filter_labels_{Path(save_name).stem}"
    if filter_col not in items.columns:
        items[filter_col] = ""

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

    if granularity == "abstract":
        prompt_file = config.prompt_files["label_abstract"]
    elif granularity == "pair":
        prompt_file = config.prompt_files["label_extracted"]
    else:
        prompt_file = config.prompt_files["label_sent_para"]
    prompt = load_formatted_prompt(prompt_file, project=project)

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
            if doc_id in processed and (row_id, pair_id) in processed[doc_id]:
                continue
        else:
            if doc_id in processed and row_id in processed[doc_id]:
                continue

        if granularity != "abstract":
            if row["classification"] != include_class:
                continue

        # Skip if filter_labels already exist
        if isinstance(row[filter_col], str) and row[filter_col].strip():
            continue

        # Rebuild context window from previous N items in same document
        start = max(i - context_length, 0)
        context_window = [
            items.iloc[j][f"{granularity}"]
            for j in range(start, i)
            if str(items.iloc[j].get("document_id", "unknown")) == doc_id
        ]

        if on_extracted:
            pair_text = f"Cause: {row.get('cause', '')}\nEffect: {row.get('effect', '')}"
            labels = handle_labelling(
                granularity=granularity,
                item=row[f"{granularity}"],
                context_window=context_window,
                prompt=prompt,
                label_list=label_list,
                config=config,
                pair=pair_text
            )
        else:
            labels = handle_labelling(
                granularity=granularity,
                item=row[f"{granularity}"],
                context_window=context_window,
                prompt=prompt,
                label_list=label_list,
                config=config
            )
        raw_labels = [label.strip() for label in labels if label.strip()]
        items.at[i, filter_col] = ", ".join(raw_labels)
        items.to_csv(output_file, index=False)

        if on_extracted:
            processed.setdefault(doc_id, []).append((row_id, pair_id))
        else:
            processed.setdefault(doc_id, []).append(row_id)
        updated_count += 1

        with open(checkpoint_file, "w") as f:
            json.dump(processed, f)

        if on_extracted:
            log_msg = f"Labeled filters for pair {pair_id} in {granularity} {row_id} (doc {doc_id}): {labels}"
        else:
            log_msg = f"Labeled filters for {granularity} {row_id} in document {doc_id}: {labels}"
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
        config (flashqda.PipelineConfig): Configuration for prompts and extractable labels.
        granularity (str, optional): Unit of analysis, "sentence" or "paragraph". Defaults to "sentence".
        context_length (int, optional): Number of prior items to include as context. Defaults to 1.
        include_class (str, optional): Classification label required for an item to be eligible for extraction.
            Defaults to the first label in `config.labels`.
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
    output_directory = Path(output_directory) if output_directory else project.results
    save_name = save_name if save_name else "extracted.csv"
    output_file = output_directory / save_name
    output_directory.mkdir(parents=True, exist_ok=True)
    include_class = include_class if include_class else config.labels[0]

    log_path = output_directory / "logs"
    log_path.mkdir(exist_ok=True)
    log_file = log_path / f"{Path(save_name).stem}.log"

    temp_path = output_directory / "temp"
    temp_path.mkdir(exist_ok=True)
    checkpoint_file = temp_path / f"{Path(save_name).stem}.checkpoint.json"

    items = pd.read_csv(input_file)
    context_window = []

    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            processed = json.load(f)
    else:
        processed = {}

    prompt_file = config.prompt_files["extract"]
    prompt = load_formatted_prompt(prompt_file, project=project)

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


        if row["classification"] == include_class: # Include items of the chosen type (e.g., "causal")
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
                relationships = response.get("relationships", [{}])

                pair_id = 1
                for rel in relationships:
                    extended_row = result_row.copy()
                    for label in config.extract_labels:
                        extended_row[label] = rel.get(label, "")
                    extended_row["pair_id"] = pair_id
                    pd.DataFrame([extended_row]).to_csv(output_file, mode='a', index=False, header=write_header)
                    write_header = False
                    pair_id += 1
                updated_count += 1
                update_log(log_file, f"Processed {granularity} {row_id} in document {doc_id}: {relationships}")
            else:
                # Append causal row with filters with empty extractions
                for label in config.extract_labels:
                    result_row[label] = ""
                result_row["pair_id"] = ""
                pd.DataFrame([result_row]).to_csv(output_file, mode='a', index=False, header=write_header)
                write_header = False
        else:
            # Append non-causal row with empty extractions
            for label in config.extract_labels:
                result_row[label] = ""
            result_row["pair_id"] = ""
            pd.DataFrame([result_row]).to_csv(output_file, mode='a', index=False, header=write_header)
            write_header = False

        processed.setdefault(doc_id, []).append(row_id)

        with open(checkpoint_file, "w") as f:
            json.dump(processed, f)

    num_docs = items["document_id"].nunique()
    print(f"Extracted from {updated_count} items in {num_docs} documents.")
    return output_file

