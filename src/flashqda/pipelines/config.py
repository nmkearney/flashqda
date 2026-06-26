from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class PipelineConfig:
    """
    Configuration for a FlashQDA analysis pipeline.

    Bundles every setting needed to run one or more pipeline stages — which
    LLM and embedding provider to use, which prompt files to load, and how
    the structured-output schemas should look. Construct with
    `PipelineConfig.from_type()` to start from a built-in preset, or
    instantiate directly and supply all fields yourself.

    Pipeline logic fields:
        pipeline_type (str, optional): Preset name (e.g. ``"causal"``,
            ``"thematic"``). Set by ``from_type()``; may be ``None`` for
            fully custom configs.
        classify_labels (List[str]): Ordered list of label strings for the
            classification step (e.g. ``["causal", "non-causal"]``).
        extract_fields (List[str]): Field names for the extraction step
            (e.g. ``["cause", "effect"]``).
        prompt_files (Dict[str, str]): Map of stage name to prompt filename,
            e.g. ``{"classify": "classify.txt", "extract": "extract.txt"}``.
        system_prompt (str): System-role message prepended to every LLM call.
        classify_schema (Dict, optional): JSON Schema for classification
            responses. Auto-generated from ``classify_labels`` when ``None``.
        extract_schema (Dict, optional): JSON Schema for extraction responses.
            Auto-generated from ``extract_fields`` when ``None``.

    LLM provider fields:
        provider (str): LLM provider — ``"openai"``, ``"anthropic"``,
            ``"ollama"``, or ``"openai_compatible"``.
        model (str): Model identifier (e.g. ``"gpt-4o-mini"``).
        prompt_mode (str): ``"auto"`` selects local prompts for Ollama/
            compatible providers; ``"cloud"`` or ``"local"`` forces a choice.
        base_url (str, optional): Base URL for ``openai_compatible`` endpoints.
        api_key (str, optional): API key override; falls back to env vars.
        temperature (float): Sampling temperature. Default ``0.0``.
        timeout (int): Per-request timeout in seconds. Default ``15``.
        max_tokens (int): Maximum tokens in the LLM response. Default ``4096``.
        use_json_mode (bool): Request structured JSON output. Default ``True``.

    Embedding fields:
        embedding_provider (str): Embedding provider — ``"openai"``,
            ``"openai_compatible"``, or ``"sentence_transformers"``.
        embedding_model (str): Embedding model identifier.
        embedding_batch_size (int): Items per batch when calling the
            embeddings API. Default ``100``.
    """

    pipeline_type: Optional[str] = None

    # --- pipeline logic ---
    classify_labels: List[str] = field(default_factory=list)
    extract_fields: List[str] = field(default_factory=list)
    prompt_files: Dict[str, str] = field(default_factory=dict)
    system_prompt: str = "You are a helpful assistant."

    # --- structured output schemas ---
    classify_schema: Optional[Dict[str, Any]] = None
    extract_schema: Optional[Dict[str, Any]] = None

    # --- LLM settings ---
    provider: str = "openai"
    model: str = "gpt-5.4-mini"
    prompt_mode: str = "auto"   # "auto" | "cloud" | "local"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0
    timeout: int = 15
    max_tokens: int = 4096
    use_json_mode: bool = True

    # --- Embeddings settings ---
    embedding_provider: str = "openai" # or "openai_compatible" | "sentence_transformers"
    embedding_model: str = "text-embedding-3-small" # or "BAAI/bge-large-en-v1.5"
    embedding_batch_size: int = 100

    @classmethod
    def from_type(cls, pipeline_type: str, topic: Optional[str] = None, **overrides):
        """
        Build a ``PipelineConfig`` from a named preset.

        Args:
            pipeline_type (str): Built-in preset name. Currently supported:
                ``"causal"``, ``"thematic"``, ``"tradeoff"``, ``"synergy"``.
            topic (str, optional): If provided, appended to the system prompt
                as ``"The topic is: <topic>."``.
            **overrides: Any ``PipelineConfig`` field can be overridden here,
                e.g. ``provider="anthropic"``, ``model="claude-opus-4-7"``,
                ``temperature=0.2``.

        Returns:
            PipelineConfig: A fully populated configuration instance.

        Raises:
            ValueError: If ``pipeline_type`` is not a recognised preset.
        """
        from flashqda.pipelines.default_configs import PIPELINE_CONFIGS

        if pipeline_type not in PIPELINE_CONFIGS:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")

        base_config = PIPELINE_CONFIGS[pipeline_type].copy()

        # Apply topic to system prompt
        if topic:
            base_prompt = base_config.get("system_prompt", "You are a helpful assistant.")
            base_config["system_prompt"] = f"{base_prompt} The topic is: {topic}."

        # Apply overrides (provider, model, base_url, timeout, etc.)
        base_config.update(overrides)

        return cls(pipeline_type=pipeline_type, **base_config)
    
    def __post_init__(self):
        if self.classify_labels and self.classify_schema is None:
            self.classify_schema = {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": list(self.classify_labels)
                    }
                },
                "required": ["label"],
                "additionalProperties": False
            }

        if self.extract_fields and self.extract_schema is None:
            self.extract_schema = {
                "type": "object",
                "properties": {
                    "relationships": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                field: {"type": "string"} for field in self.extract_fields
                            },
                            "required": list(self.extract_fields),
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["relationships"],
                "additionalProperties": False
            }