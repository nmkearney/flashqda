# Changelog

All notable changes to FlashQDA are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.2.2] — 2026

### Added
- **Anthropic Claude provider**: `provider="anthropic"` with `ANTHROPIC_API_KEY`; `max_tokens` field on `PipelineConfig`.
- **Modern embedding system**: unified `embeddings/provider.py` + `embeddings/cache.py`; batched API calls; new on-disk format `{"model": "...", "embeddings": {"text": [floats]}}`.
- **`group_items()`**: AHC-based semantic clustering with LLM-generated category labels, dendrogram export, and checkpoint/resume.
- **`estimate_cost()`**: token-count and cost estimation across all pipeline stages; no hardcoded pricing — caller supplies rates.
- **pySBD sentence segmenter**: replaces NLTK; 97.9 % accuracy on the Golden Rule Set; handles `Dr.`, `Fig.`, `et al.`, `approx.`, `ca.`, etc.
- **PDF and DOCX ingestion**: `document_io.py` supports `.txt`, `.pdf` (PyMuPDF), and `.docx` (python-docx).
- **Pipeline presets**: `"thematic"`, `"tradeoff"`, and `"synergy"` types added to `PIPELINE_CONFIGS`.
- **Local prompt variants**: `src/flashqda/prompts/local/` with simplified prompts for Ollama / OpenAI-compatible providers; `prompt_mode` field on `PipelineConfig` (`"auto"` | `"cloud"` | `"local"`).
- **Column auto-injection**: any CSV missing `document_id` or `*_id` columns gets row-based IDs assigned automatically with a console notice.
- **`analyses/` project structure**: `initialize_project` creates `analyses/`; `ProjectContext.analysis_dir(type)` generates timestamped run directories; all pipeline functions default `output_directory` there.
- **Column validation**: `pipelines/validation.py` with `validate_columns()` called in `classify_items`, `label_items`, `extract_from_classified`, `refine_extracted`.
- **Test suite**: 23+ tests across `tests/test_public_api.py`, `test_config.py`, `test_cache.py`, `test_imports.py`, `test_prompt_loader.py`, `test_preprocessing.py`, `test_cost_estimation.py`.
- **Cluster-labelling prompts** extracted to `src/flashqda/prompts/label_cluster.txt` and `label_cluster_tradeoff.txt`; user overrides supported via `project.prompts/`.
- **Optional dependency groups** in `pyproject.toml`: `grouping`, `anthropic`, `preprocessing`, `dev`.
- **`scipy`, `scikit-learn`, `matplotlib`** added to core dependencies.

### Changed
- `embed_items()` and `link_items()` migrated to the modern embedding system.
- `preprocess_documents()` sentence segmenter replaced with pySBD; paragraph segmenter splits on blank lines; PDF line-wrap artifacts collapsed before segmentation.
- `tqdm` import changed to `tqdm.auto` throughout `pipeline_runner.py` for notebook and terminal compatibility.
- `PipelineConfig.extract_fields` is the correct attribute name (was incorrectly referenced as `extract_labels` in two places — fixed).

### Removed
- **`openai_utils.py`** — dead code, fully superseded by `llm_utils.py`.
- **`embedding_core.py`**, **`embedding_cache.py`**, **`embedding_analysis.py`** — legacy embedding modules replaced by `embeddings/`.
- **`"abstract"` granularity** — retired; replaced by generic ID auto-injection so any CSV granularity is supported.
- **`label_abstract.txt`** prompt files removed.

### Fixed
- `__init__.py` typo: `expode_extractions` → `explode_extractions` in `__all__`; `refine_extracted` added to `__all__`.
- `embeddings/cache.py` `get_embeddings()` now raises `KeyError` for missing texts instead of silently dropping them.

---

## [1.2.1] — 2025

### Added
- **OpenAI-compatible provider**: `provider="openai_compatible"` with `base_url` parameter supports LM Studio, vLLM, and other OpenAI-API-compatible local servers.

### Changed
- Updated OpenAI SDK compatibility.

---

## [1.2.0] — 2025

### Changed
- **Major refactor**: codebase modularised into focused files (`pipeline_runner.py`, `llm_utils.py`, `embedding_pipeline.py`, `causal_chain.py`); class-based architecture for config and context.
- `PipelineConfig` dataclass introduced to carry all pipeline, LLM, and embedding settings.
- `ProjectContext` class introduced to manage project paths.
- Provider registry pattern in `llm_utils.py`; `send_to_llm()` dispatches to pluggable provider functions.

---

## [1.1.1] — 2024

### Fixed
- Bug fixes in labelling pipeline.

### Changed
- QuickStart notebook updated with revised examples.

---

## [1.1.0] — 2024

### Added
- Labelling on unclassified paragraphs and sentences (not only classified items).
- API key retrieval integrated into `initialize_project`.

### Fixed
- Cross-platform log file compatibility (Windows line endings).

### Changed
- Updated OpenAI SDK compatibility.

---

## [1.0.0] — 2024

### Added
- Initial release.
- Core pipeline: `preprocess_documents`, `classify_items`, `label_items`, `extract_from_classified`, `group_items`, `link_items`.
- OpenAI provider only (`LLM_API_KEY` env var or `llm_api_key.txt` file).
- Checkpoint-driven resumability for all long-running stages.
- Progressive CSV output (incremental writes, not held in memory).
- Prompt customisation via `project.prompts/` directory.
