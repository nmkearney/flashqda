"""Tests for EmbeddingCache behaviour."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from flashqda.embeddings.cache import EmbeddingCache


def _make_cache(tmpdir, texts_and_vectors):
    cache = EmbeddingCache(Path(tmpdir) / "test_cache.json", "test-model")
    for text, vec in texts_and_vectors.items():
        cache.data[text] = vec
    return cache


def test_get_embeddings_returns_correct_order():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = _make_cache(tmpdir, {"a": [1.0, 0.0], "b": [0.0, 1.0]})
        result = cache.get_embeddings(["b", "a"])
        assert result[0].tolist() == [0.0, 1.0]
        assert result[1].tolist() == [1.0, 0.0]


def test_get_embeddings_raises_on_missing_text():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = _make_cache(tmpdir, {"text1": [0.1, 0.2], "text2": [0.3, 0.4]})
        with pytest.raises(KeyError):
            cache.get_embeddings(["text1", "text_missing"])


def test_get_embeddings_raises_when_all_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = _make_cache(tmpdir, {"a": [1.0, 0.0]})
        with pytest.raises(KeyError):
            cache.get_embeddings(["not_here"])


def test_has_all_true_when_present():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = _make_cache(tmpdir, {"x": [1.0], "y": [2.0]})
        assert cache.has_all(["x", "y"])


def test_has_all_false_when_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = _make_cache(tmpdir, {"x": [1.0]})
        assert not cache.has_all(["x", "z"])


def test_get_missing_texts():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = _make_cache(tmpdir, {"a": [1.0], "b": [2.0]})
        missing = cache.get_missing_texts(["a", "c", "d"])
        assert missing == ["c", "d"]


def test_add_embeddings_and_retrieve():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache.json"
        cache = EmbeddingCache(cache_path, "my-model")
        texts = ["hello", "world"]
        vectors = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        cache.add_embeddings(texts, vectors)

        result = cache.get_embeddings(["world", "hello"])
        assert result[0].tolist() == pytest.approx([0.3, 0.4])
        assert result[1].tolist() == pytest.approx([0.1, 0.2])


def test_model_mismatch_resets_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache.json"
        cache1 = EmbeddingCache(cache_path, "model-a")
        cache1.data["text"] = [1.0, 2.0]
        cache1.save_cache()

        cache2 = EmbeddingCache(cache_path, "model-b")
        assert "text" not in cache2.data


def test_save_and_reload():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache.json"
        cache1 = EmbeddingCache(cache_path, "my-model")
        cache1.data["hello"] = [0.5, 0.6]
        cache1.save_cache()

        cache2 = EmbeddingCache(cache_path, "my-model")
        assert "hello" in cache2.data
        assert cache2.data["hello"] == [0.5, 0.6]
