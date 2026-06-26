# FlashQDA
[![DOI](https://zenodo.org/badge/916063558.svg)](https://doi.org/10.5281/zenodo.19070867)

FlashQDA is a Python package for LLM-powered qualitative data analysis (QDA). It provides a structured, resumable pipeline of functions that take a corpus of text documents and progressively classify, label, extract, group, and link the concepts within them. It is designed for academic researchers, policy analysts, and social scientists working with interview transcripts, policy documents, or published literature.

## Installation

```bash
pip install flashqda
```

Optional dependency groups:

```bash
pip install "flashqda[grouping]"        # semantic grouping (group_items)
pip install "flashqda[preprocessing]"   # PDF and DOCX support
pip install "flashqda[anthropic]"       # Anthropic Claude provider
```

## What it does

| Step | Function | Purpose |
|------|----------|---------|
| 1 | `preprocess_documents` | Segment `.txt`/`.pdf`/`.docx` files into sentences or paragraphs |
| 2 | `classify_items` | LLM binary/multiclass classification (e.g. causal / non-causal) |
| 3 | `label_items` | Apply user-defined filter tags at any stage |
| 4 | `extract_from_classified` | Structured extraction of relationships (e.g. cause/effect pairs) |
| 5 | `refine_extracted` | LLM validation and correction of extracted pairs |
| 6 | `explode_extractions` | Normalize from one row per source to one row per extracted item |
| 7 | `embed_items` | Generate and cache vector embeddings |
| 8 | `group_items` | AHC semantic clustering with LLM-generated category labels |
| 9 | `link_items` | Cosine-similarity linking of semantically related items |

Built-in pipeline presets: `"causal"`, `"thematic"`, `"tradeoff"`, `"synergy"`.

## Quick example

```python
import flashqda

# Set up project
flashqda.initialize_project("/path/to/my_project")
project = flashqda.ProjectContext("/path/to/my_project")
config = flashqda.PipelineConfig.from_type("causal")

# Run the pipeline
flashqda.preprocess_documents(project=project, granularity="sentence")

classified = flashqda.classify_items(
    project=project, config=config,
    input_file=project.data / "sentence.csv"
)

extracted = flashqda.extract_from_classified(
    project=project, config=config, input_file=classified
)

exploded = flashqda.explode_extractions(project=project, config=config, input_file=extracted)

embedding_file = flashqda.embed_items(project=project, config=config, input_file=exploded)

flashqda.link_items(project=project, config=config,
                    input_file=exploded, embedding_file=embedding_file)
```

## Supported providers

**LLM:** `openai` (default), `anthropic`, `ollama` (local), `openai_compatible`

**Embeddings:** `openai`, `sentence_transformers`, `openai_compatible`

Sensitive-data workflows are fully supported via local Ollama models — no API key required.

## Documentation

Full walkthrough with all functions and examples: [docs/QUICK_START.md](docs/QUICK_START.md)

API reference: `help(flashqda.classify_items)` (all public functions have docstrings)

## Citation

If you use FlashQDA in your research, please cite:

> Kearney, N. (2026). flashQDA (v1.2.2). https://github.com/nmkearney/flashqda

## Links

- GitHub: [https://github.com/nmkearney/flashqda](https://github.com/nmkearney/flashqda)
- PyPI: [https://pypi.org/project/flashqda](https://pypi.org/project/flashqda)
- Issues: [https://github.com/nmkearney/flashqda/issues](https://github.com/nmkearney/flashqda/issues)
