"""Tests for PipelineConfig dataclass behaviour."""
import pytest
from flashqda import PipelineConfig


def test_classify_schema_auto_generated():
    config = PipelineConfig(classify_labels=["a", "b"])
    assert config.classify_schema is not None
    assert set(config.classify_schema["properties"]["label"]["enum"]) == {"a", "b"}


def test_extract_schema_auto_generated():
    config = PipelineConfig(extract_fields=["cause", "effect"])
    assert config.extract_schema is not None
    props = config.extract_schema["properties"]["relationships"]["items"]["properties"]
    assert "cause" in props
    assert "effect" in props


def test_from_type_causal():
    config = PipelineConfig.from_type("causal", topic="agriculture")
    assert config.pipeline_type == "causal"
    assert config.classify_labels == ["causal", "non-causal"]
    assert config.extract_fields == ["cause", "effect"]
    assert "agriculture" in config.system_prompt


def test_extract_fields_attribute_not_extract_labels():
    config = PipelineConfig.from_type("causal")
    assert hasattr(config, "extract_fields"), "PipelineConfig must have extract_fields"
    assert not hasattr(config, "extract_labels"), "extract_labels does not exist on PipelineConfig"


def test_unknown_pipeline_type_raises():
    with pytest.raises(ValueError, match="Unknown pipeline type"):
        PipelineConfig.from_type("nonexistent_type")


def test_schema_not_overwritten_when_explicit():
    custom_schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    config = PipelineConfig(classify_labels=["a", "b"], classify_schema=custom_schema)
    assert config.classify_schema is custom_schema
