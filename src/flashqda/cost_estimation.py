import json
from pathlib import Path
from typing import Optional

import pandas as pd

from flashqda.pipelines.config import PipelineConfig

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Rough output token budget per stage (JSON response body)
_OUTPUT_TOKENS = {
    "classify": 30,
    "label_sent_para": 60,
    "label_extracted": 60,
    "extract": 120,
    "refine_extract": 150,
}


def _count_tokens(text: str, model: str) -> int:
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return max(1, len(text) // 4)


def _load_prompt_text(prompt_file: str) -> str:
    path = _PROMPTS_DIR / prompt_file
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def estimate_cost(
    df: pd.DataFrame,
    config: PipelineConfig,
    granularity: str = "sentence",
    input_cost_per_1m: float = 0.0,
    output_cost_per_1m: float = 0.0,
    context_length: int = 1,
) -> pd.DataFrame:
    """
    Estimate token usage and API cost for each configured pipeline stage.

    Args:
        df: DataFrame of preprocessed items (output of preprocess_documents).
        config: PipelineConfig specifying model, prompt files, and schemas.
        granularity: Column name used as the text item — "sentence" or "paragraph".
        input_cost_per_1m: USD cost per 1 million input tokens. Check your
            provider's current rate card (e.g. gpt-4o input = $2.50/1M).
        output_cost_per_1m: USD cost per 1 million output tokens (e.g. gpt-4o
            output = $10.00/1M).
        context_length: Number of prior items included as context per LLM call.
            Should match the value you plan to pass to classify_items / label_items.

    Returns:
        DataFrame with columns: stage, items, input_tokens, output_tokens,
        total_tokens, estimated_cost_usd. The final row is a TOTAL.

    Notes:
        All stages treat every item in df as processed (upper bound). In
        practice, classify filters items before extract/refine_extract runs.
        Set input_cost_per_1m and output_cost_per_1m to 0.0 to see token
        counts only (useful for local/Ollama runs with no per-token charge).
    """
    granularity = granularity if granularity in ("sentence", "paragraph") else "sentence"
    model = config.model
    num_items = len(df)

    system_tokens = _count_tokens(config.system_prompt, model)

    if granularity in df.columns and num_items > 0:
        avg_item_tokens = int(df[granularity].dropna().apply(len).mean() // 4) or 1
    else:
        avg_item_tokens = 50

    avg_context_tokens = avg_item_tokens * context_length

    classify_schema_tokens = _count_tokens(
        json.dumps(config.classify_schema or {}), model
    )
    extract_schema_tokens = _count_tokens(
        json.dumps(config.extract_schema or {}), model
    )

    rows = []
    for stage, prompt_file in config.prompt_files.items():
        prompt_text = _load_prompt_text(prompt_file)
        prompt_base_tokens = _count_tokens(prompt_text, model)

        schema_tokens = (
            extract_schema_tokens
            if stage in ("extract", "refine_extract")
            else classify_schema_tokens
        )

        input_per_item = (
            system_tokens
            + prompt_base_tokens
            + avg_item_tokens
            + avg_context_tokens
            + schema_tokens
        )
        output_per_item = _OUTPUT_TOKENS.get(stage, 60)

        total_input = input_per_item * num_items
        total_output = output_per_item * num_items
        total_tokens = total_input + total_output
        cost = (total_input * input_cost_per_1m + total_output * output_cost_per_1m) / 1_000_000

        rows.append({
            "stage": stage,
            "items": num_items,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(cost, 4),
        })

    result = pd.DataFrame(rows)

    if result.empty:
        total_row = {
            "stage": "TOTAL", "items": num_items,
            "input_tokens": 0, "output_tokens": 0,
            "total_tokens": 0, "estimated_cost_usd": 0.0,
        }
    else:
        total_row = {
            "stage": "TOTAL",
            "items": num_items,
            "input_tokens": result["input_tokens"].sum(),
            "output_tokens": result["output_tokens"].sum(),
            "total_tokens": result["total_tokens"].sum(),
            "estimated_cost_usd": round(result["estimated_cost_usd"].sum(), 4),
        }

    return pd.concat([result, pd.DataFrame([total_row])], ignore_index=True)
