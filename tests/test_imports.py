"""Tests that verify correct import choices across the package."""
import inspect


def test_pipeline_runner_uses_tqdm_auto():
    import flashqda.pipeline_runner as pr
    src = inspect.getsource(pr)
    assert "tqdm.notebook" not in src, (
        "pipeline_runner must use tqdm.auto (not tqdm.notebook) "
        "so it works outside Jupyter notebooks"
    )


def test_pipeline_runner_importable_outside_notebook():
    import flashqda.pipeline_runner  # noqa: F401  — must not raise ImportError


def test_embedding_pipeline_uses_modern_system():
    import flashqda.embedding_pipeline as ep
    src = inspect.getsource(ep)
    assert "embedding_core" not in src, "embedding_pipeline must not use the legacy embedding_core"
    assert "embedding_cache" not in src, "embedding_pipeline must not use the legacy embedding_cache"
    assert "embed_texts" in src, "embedding_pipeline must use the modern embed_texts()"


def test_causal_chain_uses_modern_cache():
    import flashqda.causal_chain as cc
    src = inspect.getsource(cc)
    assert "embedding_cache" not in src, "causal_chain must not use the legacy embedding_cache"
    assert "EmbeddingCache" in src, "causal_chain must use the modern EmbeddingCache"
