"""Tests for prompt_loader.py — local prompt selection and file resolution."""
import pytest
from unittest.mock import MagicMock
from flashqda.prompt_loader import _should_use_local_prompts, load_formatted_prompt


def _config(provider="openai", prompt_mode="auto"):
    cfg = MagicMock()
    cfg.provider = provider
    cfg.prompt_mode = prompt_mode
    return cfg


class TestShouldUseLocalPrompts:
    def test_none_config_returns_false(self):
        assert _should_use_local_prompts(None) is False

    def test_openai_auto_returns_false(self):
        assert _should_use_local_prompts(_config("openai", "auto")) is False

    def test_anthropic_auto_returns_false(self):
        assert _should_use_local_prompts(_config("anthropic", "auto")) is False

    def test_ollama_auto_returns_true(self):
        assert _should_use_local_prompts(_config("ollama", "auto")) is True

    def test_openai_compatible_auto_returns_true(self):
        assert _should_use_local_prompts(_config("openai_compatible", "auto")) is True

    def test_cloud_mode_overrides_ollama(self):
        assert _should_use_local_prompts(_config("ollama", "cloud")) is False

    def test_local_mode_overrides_openai(self):
        assert _should_use_local_prompts(_config("openai", "local")) is True


class TestLoadFormattedPromptLocalResolution:
    def test_cloud_config_loads_default(self):
        cfg = _config("openai", "auto")
        result = load_formatted_prompt(
            "causal_classify.txt",
            config=cfg,
            granularity="sentence",
            context_window="",
            item="Test item.",
            json_schema="{}",
        )
        assert "Common language cues" in result  # present in cloud prompt, not local

    def test_local_config_loads_local(self):
        cfg = _config("ollama", "auto")
        result = load_formatted_prompt(
            "causal_classify.txt",
            config=cfg,
            granularity="sentence",
            context_window="",
            item="Test item.",
            json_schema="{}",
        )
        assert "Common language cues" not in result  # stripped from local prompt
        assert "Causal:" in result

    def test_local_mode_overrides_openai_provider(self):
        cfg = _config("openai", "local")
        result = load_formatted_prompt(
            "label_cluster.txt",
            config=cfg,
            bullet_list="- item one\n- item two",
        )
        assert "title case" in result  # both versions have this phrase
        assert "Specific and descriptive" not in result  # removed from local version

    def test_no_config_loads_default(self):
        result = load_formatted_prompt(
            "label_cluster.txt",
            bullet_list="- item one",
        )
        assert "Specific and descriptive" in result  # cloud default

    def test_missing_prompt_raises(self):
        with pytest.raises(FileNotFoundError):
            load_formatted_prompt("nonexistent_prompt.txt")
