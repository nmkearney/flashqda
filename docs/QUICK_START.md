# FlashQDA Quick Start

FlashQDA is a Python package for LLM-powered qualitative data analysis (QDA). It provides a structured pipeline of functions that take a corpus of text documents and progressively classify, label, extract, group, and link the concepts within them. The default pipeline is designed for causal analysis (identifying cause-effect relationships), but built-in presets also support thematic, tradeoff, and synergy analyses.

---

## 1. Installation

**Recommended: Conda environment**

```bash
conda create --name flashqda-env python=3.10 -y
conda activate flashqda-env
pip install flashqda
```

Install optional dependencies for semantic grouping (`group_items`):

```bash
pip install "flashqda[grouping]"
```

To run the pipeline in a Jupyter notebook, register the environment as a kernel:

```bash
pip install jupyter
python -m ipykernel install --user --name=flashqda-env --display-name "flashqda-env"
jupyter notebook
```

**Input file requirements**

Place your source documents in the project `data/` folder (created below). Supported formats:
- `.txt` — must be UTF-8 encoded; check for hidden carriage returns
- `.pdf` — text extraction is layout-aware (requires `pip install "flashqda[preprocessing]"`)
- `.docx` — requires `pip install "flashqda[preprocessing]"`

---

## 2. Local LLM setup (optional)

By default FlashQDA calls OpenAI's API. For sensitive data, or to avoid API costs, you can run entirely locally using [Ollama](https://ollama.com).

```bash
# macOS
brew install ollama

# Windows
winget install Ollama.Ollama

# Pull a model (Mistral is a good general-purpose choice)
ollama pull mistral:instruct

# Start the server — leave this terminal open
ollama serve
```

Then configure FlashQDA to use it (see §4 below).

---

## 3. Project setup

```python
import flashqda

project_root = "/path/to/my_project"
flashqda.initialize_project(project_root)
# Creates: my_project/data/, my_project/analyses/, my_project/prompts/

project = flashqda.ProjectContext(project_root)
# project.data      — input documents
# project.analyses  — timestamped pipeline outputs (auto-created per run)
# project.prompts   — optional prompt overrides
```

Place your `.txt`, `.pdf`, or `.docx` files in `project.data` before running the pipeline.

---

## 4. Configuration

### Built-in pipeline types

```python
from flashqda import PipelineConfig

# Causal analysis — identifies cause-effect relationships
config = PipelineConfig.from_type("causal")

# Thematic analysis — identifies themes and supporting evidence
config = PipelineConfig.from_type("thematic")

# Tradeoff analysis — identifies gains and losses
config = PipelineConfig.from_type("tradeoff")

# Synergy analysis — identifies mutually reinforcing factors
config = PipelineConfig.from_type("synergy")
```

All built-in types default to `provider="openai"` and `model="gpt-4o"`.

### Switching providers

**OpenAI (default)**

Set the environment variable `LLM_API_KEY` to your OpenAI API key, or pass it directly:

```python
config = PipelineConfig.from_type("causal", api_key="sk-...")
```

**Anthropic Claude**

```python
config = PipelineConfig.from_type("causal",
    provider="anthropic",
    model="claude-opus-4-5",
    api_key="..."   # or set ANTHROPIC_API_KEY env var
)
```

**Local Ollama**

```python
config = PipelineConfig.from_type("causal",
    provider="ollama",
    model="mistral:instruct",
    base_url="http://localhost:11434/v1"
)
```

### Custom pipeline config

For an analysis type not covered by the built-in presets, define a `PipelineConfig` directly and place matching prompt files in `project.prompts/`:

```python
config = flashqda.PipelineConfig(
    classify_labels=["tradeoff", "not_tradeoff"],
    extract_fields=["gain", "cost"],
    prompt_files={
        "classify":        "tradeoff_classify.txt",
        "label_sent_para": "label_sent_para.txt",
        "label_extracted": "label_extracted.txt",
        "extract":         "tradeoff_extract.txt",
        "refine_extract":  "refine_tradeoff_extract.txt",
    },
    system_prompt="You are helping identify tradeoffs in scientific text.",
    provider="openai",
    model="gpt-4o",
)
```

---

## 5. Pipeline walkthrough

The example below uses the causal preset on two agroforestry articles. Sample files (`Lojka et al. 2016.txt`, `Ocampo-Ariza et al. 2023.txt`) are in the `docs/` folder of the GitHub repository.

### Checkpointing and resuming

Every LLM stage (`classify_items`, `label_items`, `extract_from_classified`, `refine_extracted`) saves progress to a JSON checkpoint file after each item. If a run is interrupted you can resume it exactly where it left off.

The key: `project.analysis_dir(...)` creates a **new** timestamped directory every time it is called. If you let the function auto-create the directory and then re-call it after an interruption, it starts a fresh run in a new directory rather than resuming. To resume, capture the output directory explicitly before the first call and pass it again on retry:

```python
# Create the output directory once and reuse it across calls
output_dir = project.analysis_dir("classification")

classified_path = flashqda.classify_items(
    project=project,
    config=config,
    granularity="sentence",
    input_file=project.data / "sentences.csv",
    output_directory=output_dir,   # explicit — safe to re-run
    save_name="classified.csv"
)
```

Re-running the same call with the same `output_directory` and `save_name` automatically reads the checkpoint and skips already-processed items. To start from scratch instead, pass a new directory or omit `output_directory` so a fresh timestamped one is created. See [troubleshooting.md](troubleshooting.md) for checkpoint file locations and how to delete one.

---

### Step 1 — Preprocess documents

Segment all files in `data/` into sentences (or paragraphs). Assigns a `document_id` to each file and a `sentence_id` to each sentence.

```python
flashqda.preprocess_documents(
    project=project,
    granularity="sentence",   # "sentence" (default) or "paragraph"
    save_name="sentences.csv" # saved to project.data/
)
```

### Step 2 — Classify items

Label each sentence as `causal` or `non-causal` using the LLM.

```python
config = flashqda.PipelineConfig.from_type("causal")

classified_path = flashqda.classify_items(
    project=project,
    config=config,
    granularity="sentence",
    context_length=1,           # number of prior sentences sent as context
    input_file=project.data / "sentences.csv",
    save_name="classified.csv"
)
```

Output: a CSV with a `classification` column added to every row.

`context_length` controls how many preceding items are passed to the LLM as background context. With sentence granularity, 1 is almost always worthwhile — sentences are short and referencing the prior sentence resolves pronouns and subject continuity at low token cost. With paragraph granularity, 0 is a reasonable starting point: paragraphs are more self-contained and including the previous paragraph adds substantially more tokens per call; increase to 1 only if your documents regularly refer back to entities or findings introduced in the preceding paragraph.

### Step 3 — Label classified items (optional)

Apply custom filter tags to the classified sentences. Useful for inclusion/exclusion criteria before extraction.

```python
label_list = [
    {"name": "substantive_not_methodological",
     "description": "The sentence discusses the topic studied, not how the study was conducted."},
    {"name": "descriptive_not_prescriptive",
     "description": "The sentence describes why something happens, not what should be done."},
    {"name": "definitive_not_ambiguous",
     "description": "The sentence states a relationship without hedging (e.g. 'may cause')."},
]

labelled_path = flashqda.label_items(
    project=project,
    config=config,
    granularity="sentence",
    context_length=1,
    include_class="causal",     # only process sentences classified as causal
    label_list=label_list,
    on_classified=True,
    expand=True,                # adds one-hot columns for each label
    input_file=classified_path,
    save_name="classified_labelled.csv"
)
```

### Step 4 — Extract from classified

Extract structured cause-effect pairs from each causal sentence.

```python
extracted_path = flashqda.extract_from_classified(
    project=project,
    config=config,
    granularity="sentence",
    context_length=1,
    include_class="causal",
    filter_column="substantive_not_methodological",  # skip rows where this is False
    filter_keys="FALSE",
    input_file=labelled_path,
    save_name="extracted.csv"
)
```

Output: a CSV with an `extractions_json` column containing a JSON list of `{"cause": ..., "effect": ...}` pairs per sentence.

### Step 5 — Refine extractions (optional)

Pass extracted pairs back through the LLM for validation and correction.

```python
refined_path = flashqda.refine_extracted(
    project=project,
    config=config,
    granularity="sentence",
    context_length=1,
    input_file=extracted_path,
    save_name="extracted_refined.csv"
)
```

### Step 6 — Explode extractions

Normalize from one row per source sentence to one row per extracted pair.

```python
exploded_path = flashqda.explode_extractions(
    project=project,
    config=config,
    granularity="sentence",
    input_file=refined_path,    # or extracted_path if skipping Step 5
    save_name="extracted_exploded.csv"
)
```

After this step each row has `cause` and `effect` columns with a single value.

### Step 7 — Label extracted pairs (optional)

Apply thematic tags to the extracted cause-effect pairs. This differs from Step 3, which labelled individual sentences: here the LLM sees the full extracted pair (e.g. both the cause and the effect) rather than just the source sentence. Pass `on_extracted=True` to activate this mode; the input must be the exploded CSV produced in Step 6, which contains a `pair_id` column alongside the extracted fields.

```python
label_list = [
    {"name": "social_system",
     "description": "The pair relates to social systems (demography, economics, politics, culture)."},
    {"name": "ecological_system",
     "description": "The pair relates to ecological systems (resources, habitats, ecosystem services)."},
    {"name": "barrier",
     "description": "The pair describes why something does not or cannot happen."},
    {"name": "driver",
     "description": "The pair describes why something does happen."},
]

labelled_pairs_path = flashqda.label_items(
    project=project,
    config=config,
    granularity="sentence",
    context_length=1,
    include_class="causal",
    label_list=label_list,
    on_extracted=True,
    expand=True,
    input_file=exploded_path,
    save_name="extracted_labelled.csv"
)
```

### Step 8 — Embed items

Generate vector embeddings for all unique causes and effects. Results are cached in a JSON file; re-running only embeds new texts.

```python
embedding_path = flashqda.embed_items(
    project=project,
    config=config,
    column_names=["cause", "effect"],
    input_file=exploded_path,
    save_name="embeddings.json"
)
```

### Step 9 — Group items (optional)

Cluster semantically similar items and generate a descriptive label for each cluster (e.g., "growth in the macroeconomy" and "overall economic expansion" → "economic growth"). Requires `pip install "flashqda[grouping]"`.

```python
flashqda.group_items(
    input_file=exploded_path,
    column_names=["cause", "effect"],
    similarity_threshold=0.6,   # higher = tighter clusters
    config=config,
    project=project,
    save_name="grouped"
)
```

Output: a CSV with a `category` column and a `categories.csv` file you can edit before remapping.

To remap after editing:

```python
flashqda.remap_from_categories_csv(
    input_file=exploded_path,
    categories_file="grouped_categories.csv",
    column_names=["cause", "effect"],
    output_file="extracted_categorised.csv"
)
```

### Step 10 — Link items

Find pairs of sentences where the effect of one is semantically similar to the cause of another. This can be used to construct a causal chain or graph.

```python
links_path = flashqda.link_items(
    project=project,
    config=config,
    threshold=0.85,             # cosine similarity cutoff (0–1)
    input_file=exploded_path,
    embedding_file=embedding_path,
    save_name="suggested_links.csv"
)
```

Output: a CSV listing `from_effect → to_cause` pairs with similarity scores and source metadata.

---

## 6. Other pipeline types

The same steps apply for other pipeline types — only the config and field names differ.

**Thematic analysis** (no classify step; goes straight to extract):

```python
config = flashqda.PipelineConfig.from_type("thematic")
# extract_fields = ["theme", "evidence"]
```

**Tradeoff analysis**:

```python
config = flashqda.PipelineConfig.from_type("tradeoff")
# classify_labels = ["tradeoff", "non-tradeoff"]
# extract_fields = ["gain", "loss"]
```

**Synergy analysis**:

```python
config = flashqda.PipelineConfig.from_type("synergy")
# classify_labels = ["synergy", "non-synergy"]
# extract_fields = ["factor_a", "factor_b"]
```

---

## 7. Estimate cost before running

Before committing to a full run, estimate token usage and cost using your preprocessed dataframe:

```python
import pandas as pd

df = pd.read_csv(project.data / "sentences.csv")
config = flashqda.PipelineConfig.from_type("causal")

cost_estimate = flashqda.estimate_cost(
    df=df,
    config=config,
    granularity="sentence",
    context_length=1,
    input_cost_per_1m=2.50,    # USD per 1M input tokens — check your provider's rate card
    output_cost_per_1m=10.00,  # USD per 1M output tokens
)
print(cost_estimate)
```

Rates for common models (as of mid-2025):

| Model | Input ($/1M) | Output ($/1M) |
|-------|-------------|---------------|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| claude-opus-4-5 | $15.00 | $75.00 |
| mistral:instruct (local) | $0.00 | $0.00 |

---

## 8. Custom prompts

All prompts can be overridden by placing a `.txt` file with the same name in `project.prompts/`. FlashQDA checks for user prompts before falling back to the library defaults. For example, to override the classification prompt for a causal pipeline:

```
my_project/
└── prompts/
    └── causal_classify.txt   ← your version
```

Prompt filenames for each stage are listed in `config.prompt_files`.

---

## 9. Getting help

All public functions have full docstrings:

```python
help(flashqda.classify_items)
help(flashqda.PipelineConfig)
```

For issues or feature requests, visit [https://github.com/nmkearney/flashqda](https://github.com/nmkearney/flashqda).
