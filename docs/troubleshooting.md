# FlashQDA Troubleshooting Guide

---

## API key setup

### OpenAI

FlashQDA looks for the API key in two places, in this order:

1. **Environment variable** — set `LLM_API_KEY` before starting Python:
   ```bash
   export LLM_API_KEY="sk-..."
   ```
2. **Key file** — place a plain-text file named `llm_api_key.txt` in your project root (one line, the key only).

You can also pass the key directly in config:
```python
config = PipelineConfig.from_type("causal", api_key="sk-...")
```

### Anthropic

Set the `ANTHROPIC_API_KEY` environment variable:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Then use `provider="anthropic"` in your config:
```python
config = PipelineConfig.from_type("causal", provider="anthropic", model="claude-opus-4-7")
```

### Ollama (local)

No API key is required. Start the Ollama server before running FlashQDA:
```bash
ollama serve
```

Then pull the model you want to use (one-time):
```bash
ollama pull mistral
```

Configure FlashQDA to use it:
```python
config = PipelineConfig.from_type("causal", provider="ollama", model="mistral")
```

Ollama defaults to `http://localhost:11434`. If your server runs on a different host or port, set `base_url`:
```python
config = PipelineConfig.from_type("causal", provider="ollama", model="mistral",
                                   base_url="http://192.168.1.10:11434")
```

### OpenAI-compatible endpoints (LM Studio, vLLM)

Pass `provider="openai_compatible"` and the server's base URL. No API key is required for most local servers:
```python
config = PipelineConfig.from_type("causal",
                                   provider="openai_compatible",
                                   model="mistral-7b-instruct",
                                   base_url="http://localhost:1234/v1")
```

---

## Progress bars (tqdm)

FlashQDA uses `tqdm.auto`, which automatically selects a notebook-style or terminal-style progress bar based on the execution context. No manual configuration is needed.

**If you see raw `\r` escape characters** in a terminal or log file, your tqdm version may be outdated. Upgrade it:
```bash
pip install --upgrade tqdm
```

**If progress bars do not appear in Jupyter** despite `tqdm.auto`, ensure the `ipywidgets` package is installed and the notebook server has been restarted after installation:
```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

---

## Checkpoint recovery

Every long-running pipeline stage (`classify_items`, `label_items`, `extract_from_classified`, `refine_extracted`, `embed_items`, `group_items`) saves progress to a JSON checkpoint file after each item is processed.

**Checkpoint location:**
```
analyses/<type>/<YYYY.MM.DD-HH-MM-SS>/temp/<save_name>.checkpoint.json
```

**To resume after an interruption**, simply call the same pipeline function again with the same `output_directory` and `save_name`. FlashQDA reads the checkpoint automatically and skips already-processed items.

**To restart from scratch**, either:
- Delete the checkpoint file, or
- Pass a new `output_directory` (a fresh timestamped directory is created automatically if you omit this parameter).

**Example — manual restart:**
```python
import os
os.remove("analyses/classification/2026.01.15-10-00-00/temp/classified.csv.checkpoint.json")
classify_items(project=project, config=config, granularity="sentence")
```

**Partial output files:** the output CSV is written incrementally. If a run was interrupted, the output file may already contain partial results. Resuming from the checkpoint will append the remaining rows correctly — do not delete the partial output CSV.

---

## Column naming requirements

FlashQDA pipeline functions expect specific column names. If your CSV uses different names, the function will raise a `ValueError` listing the missing columns.

### Required columns by stage

| Stage | Required columns |
|-------|-----------------|
| `classify_items` | `document_id`, `sentence_id` or `paragraph_id`, `sentence` or `paragraph` |
| `label_items` | `document_id`, `sentence_id` or `paragraph_id`, `sentence` or `paragraph` |
| `extract_from_classified` | `document_id`, `sentence_id` or `paragraph_id`, `sentence` or `paragraph` |
| `refine_extracted` | `document_id`, `sentence_id` or `paragraph_id`, `sentence` or `paragraph` |
| `embed_items` | columns named in `column_names` parameter |
| `group_items` | columns named in `column_names` parameter |
| `link_items` | columns named in `column_names` parameter |

### Auto-injection of IDs

If your CSV is missing `document_id` or the granularity ID column (`sentence_id` / `paragraph_id`), FlashQDA will assign row-based integer IDs automatically and print a notice:

```
[NOTICE] 'document_id' not found — assigning sequential IDs.
```

This is intentional for workflows that start from an ad-hoc CSV rather than `preprocess_documents` output.

### Common mistake: non-standard column names

If you have a column called `text` instead of `sentence`, rename it before calling the pipeline function:

```python
df = df.rename(columns={"text": "sentence"})
df.to_csv("sentences.csv", index=False)
```

Or pass the renamed DataFrame directly to the function via `input_file` after saving.

### Columns produced by `preprocess_documents`

Using `preprocess_documents` to ingest your documents guarantees the correct schema:

| Column | Description |
|--------|-------------|
| `document_id` | Integer ID unique to each source file |
| `filename` | Original filename |
| `sentence_id` or `paragraph_id` | Integer ID within the document |
| `sentence` or `paragraph` | The text segment |
