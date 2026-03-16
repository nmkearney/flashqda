# labeller.py

from typing import Dict, List, Optional
import json
import time
from pathlib import Path
from tqdm import tqdm

import traceback


########################
# Helpers: logging / I/O
########################

def _log(log_path: Path, message: str):
    """
    Append a message to the pipeline log file.
    Safe even if log_path is None.
    """
    if log_path is None:
        return
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def _load_checkpoint(checkpoint_path: Optional[Path]) -> Dict[int, str]:
    """
    If we previously saved partial labels to disk, load and return them.
    Otherwise return empty dict.
    """
    if checkpoint_path is None:
        return {}
    if not Path(checkpoint_path).exists():
        return {}
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # keys in JSON will be strings, convert back to int
        return {int(k): v for k, v in data.items()}
    except Exception:
        # If checkpoint is corrupted, ignore it (don't crash pipeline)
        return {}


def _save_checkpoint(checkpoint_path: Optional[Path], labels: Dict[int, str]):
    """
    Persist partial labels to disk so we can resume later.
    """
    if checkpoint_path is None:
        return
    try:
        serializable = {str(k): v for k, v in labels.items()}
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
    except Exception:
        # Don't hard-crash on checkpoint failure, just continue.
        pass


########################
# Core LLM call helper
########################

def _build_prompt_for_cluster(cluster_items: List[str]) -> str:
    """
    Construct a prompt that asks the model to create a short, human-usable theme label.
    We DO NOT want a paragraph. We want something like:
      - "Staffing shortages / burnout"
      - "Communication gaps with management"
      - "IT-related technical failures"
    """
    # We cap the examples we feed the model so context doesn't explode for huge clusters.
    # We'll take up to, say, 20 representative items.
    MAX_EXAMPLES = 20
    examples = cluster_items[:MAX_EXAMPLES]

    bullet_list = "\n".join(f"- {item}" for item in examples)

    prompt = (
        "You are assisting in qualitative thematic analysis.\n"
        "You are given a set of semantically similar items.\n\n"
        "Task:\n"
        "1. Infer the central shared theme.\n"
        "2. Respond with ONLY a short label (max ~7 words), no punctuation at the end.\n"
        "3. The label should be specific and human-readable, not generic.\n"
        "4. Do not include quotes, numbering, or explanations.\n\n"
        "Examples from this cluster:\n"
       f"{bullet_list}\n\n"
        "Label:"
    )

    #prompt = (
    #    "You are assisting in qualitative thematic analysis.\n"
    #    "You are given a set of semantically similar trade-offs.\n\n"
    #    "Task:\n"
    #    "1. Describe the general theme of the set.\n"
    #    "2. Give the set a descriptive label based on its general theme.\n"
    #    "3. Format the label as 'X vs. Y', where X is the thing that is gained or improved and Y is the thing that is lost or worsens.\n"
    #    "4. Respond with ONLY a short label (max ~10 words), no punctuation at the end.\n"
    #    "5. The label should be specific and human-readable, not generic.\n"
    #    "6. Do not include quotes, numbering, or explanations.\n\n"
    #    "Examples from this cluster:\n"
    #    f"{bullet_list}\n\n"
    #    "Label:"
    #)

    return prompt


def _call_llm_for_label(
    client,
    model_name: str,
    cluster_items: List[str],
) -> Optional[str]:
    """
    Ask the LLM for a label for this cluster. Returns the raw string,
    or None if we couldn't get one.
    """
    prompt = _build_prompt_for_cluster(cluster_items)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        candidate = response.choices[0].message.content.strip()

        # Newer API style: response.output[0].content[0].text
        if hasattr(response, "output"):
            try:
                candidate = response.output[0].content[0].text.strip()
            except Exception:
                candidate = None

        # Fallback: response.choices[0].message.content
        if candidate is None and hasattr(response, "choices"):
            try:
                candidate = response.choices[0].message.content.strip()
            except Exception:
                candidate = None

        # Final fallback: just str(response)
        if candidate is None:
            candidate = str(response).strip()

        # Post-process: clean up quotes or trailing punctuation
        if candidate:
            candidate = candidate.strip()
            # remove wrapping quotes
            if (
                (candidate.startswith('"') and candidate.endswith('"'))
                or (candidate.startswith("'") and candidate.endswith("'"))
            ):
                candidate = candidate[1:-1].strip()
            # drop trailing period, colon, dash
            while candidate and candidate[-1] in ".:-;":
                candidate = candidate[:-1].strip()

        return candidate or None

    except Exception:
        # swallow the exception and return None so pipeline can fallback
        return None


########################
# Public API
########################

def label_clusters(
    clusters: Dict[int, List[str]],
    model_name: str,
    log_path: Optional[Path],
    checkpoint_path: Optional[Path] = None,
    max_retries_per_cluster: int = 3,
    retry_delay_seconds: float = 1.5,
    label_min_items: int = 2,
    config=None,
) -> Dict[int, str]:
    """
    Produce human-readable labels for each cluster using an LLM.

    Args:
        clusters:
            {cluster_id: [item_text, ...]}
        model_name:
            Which LLM to use (e.g. 'gpt-4o-mini').
        log_path:
            Log file path for audit trail.
        checkpoint_path:
            Where to save partial label JSON for resume safety.
        max_retries_per_cluster:
            Retry attempts for transient failures.
        retry_delay_seconds:
            Delay between retries.
        label_min_items:
            Minimum number of items required to call the LLM.
            Clusters smaller than this get a default label ("Category_<id>").
            Default = 2.

    Returns:
        labels: {cluster_id: "Readable label"}.
    """

    # 1. Load checkpoint if any
    labels: Dict[int, str] = _load_checkpoint(checkpoint_path)

    # 2. Initialize LLM client
    try:
        from flashqda.llm_utils import make_openai_client
        client = make_openai_client(config)
    except Exception as e:
        _log(log_path, f"[ERROR] Could not init LLM client: {e}")
        client = None

    for cid, items in tqdm(clusters.items(),
                           total = len(clusters),
                           desc = "Labelling clusters"):
        # Skip if already labeled from checkpoint
        if cid in labels and labels[cid]:
            continue

        # 🧠 NEW: Skip small clusters below threshold
        if len(items) < label_min_items:
            labels[cid] = f"Category_{cid}"
            _log(
                log_path,
                f"[INFO] Skipping LLM labeling for cluster {cid} "
                f"(size={len(items)} < {label_min_items}); "
                f"using default label '{labels[cid]}'",
            )
            continue

        cluster_preview = "; ".join(items[:3])
        _log(log_path, f"[INFO] Labelling cluster {cid} (size={len(items)}; preview: {cluster_preview})")

        label_text = None
        if client is not None:
            for attempt in range(1, max_retries_per_cluster + 1):
                label_text = _call_llm_for_label(client, model_name, items)
                if label_text:
                    break
                _log(log_path, f"[WARN] Attempt {attempt} failed for cluster {cid}, retrying...")
                time.sleep(retry_delay_seconds)

        # Fallback to Category_<id> if LLM failed
        if not label_text:
            label_text = f"Category_{cid}"
            _log(log_path, f"[ERROR] Falling back to default label for cluster {cid}: {label_text}")
        else:
            _log(log_path, f"[INFO] Cluster {cid} labeled as '{label_text}'")

        labels[cid] = label_text
        _save_checkpoint(checkpoint_path, labels)

    _log(log_path, "[INFO] Finished labelling all clusters.")
    return labels
