# config.py — defines reusable pipeline configurations for classification/extraction

from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class PipelineConfig:
    """
    Configuration object for labelling, classification, extraction, and grouping pipelines.

    Attributes:
        labels (List[str]): List of classification labels.
        extract_labels (List[str]): List of fields to extract (e.g., ["cause", "effect"]).
        prompt_files (Dict[str, str]): Dictionary of named prompt template paths 
            (e.g., {"classify": "path/to/classify_prompt.txt"}).
        system_prompt (str): Optional system-level prompt prefix. Defaults to a generic assistant prompt.
    """

    # --- pipeline logic ---
    labels: List[str]
    extract_labels: List[str]
    prompt_files: Dict[str, str]
    system_prompt: str = "You are a helpful assistant."

    # --- LLM settings ---
    provider: str = "openai"
    model: str = "gpt-4o"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0
    timeout: int = 15

    @classmethod
    def from_type(cls, pipeline_type: str, topic: str = None):
        """
        Create a PipelineConfig from a predefined type and optionally inject a topic.

        Looks up a base configuration from `PIPELINE_CONFIGS` and optionally extends
        the system prompt with a topic-specific phrase.

        Args:
            pipeline_type (str): The type of pipeline to load (must exist in `PIPELINE_CONFIGS`).
            topic (str, optional): Topic to inject into the system prompt for contextualization.

        Returns:
            PipelineConfig: A fully populated pipeline configuration instance.

        Raises:
            ValueError: If the given `pipeline_type` is not found in the predefined configs.
        """
        from flashqda.pipelines.default_configs import PIPELINE_CONFIGS
        if pipeline_type not in PIPELINE_CONFIGS:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        base_config = PIPELINE_CONFIGS[pipeline_type].copy()

        # Inject topic if provided
        if topic:
            base_prompt = base_config.get("system_prompt", cls.system_prompt)
            base_config["system_prompt"] = f"{base_prompt} The topic is: {topic}."

        return cls(pipeline_type=pipeline_type, **base_config)