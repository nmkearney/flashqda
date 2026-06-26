# flashqda/embeddings/provider.py
"""
Embedding provider for FlashQDA.
Handles embedding generation using LLM,
with caching, logging, and safe retry behavior.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from flashqda.embeddings.cache import EmbeddingCache
from flashqda.llm_utils import make_embedding_client
from flashqda.log_utils import update_log

def _get_embedding_provider(config) -> str:
    return getattr(config, "embedding_provider", "openai")


def _get_embedding_model(config, fallback: str) -> str:
    return getattr(config, "embedding_model", None) or fallback


def _embed_with_sentence_transformers(
    texts: List[str],
    model: str,
    batch_size: int = 64,
    log_path: Optional[Path] = None,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for local embeddings. "
            "Install it with `pip install sentence-transformers`."
        ) from e

    if log_path:
        update_log(log_path, f"[INFO] Loading local embedding model: {model}")

    embedder = SentenceTransformer(model)

    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return embeddings.astype(np.float32)


def _embed_with_openai_compatible(
    texts: List[str],
    model: str,
    cache: Optional[EmbeddingCache],
    log_path: Optional[Path],
    batch_size: int,
    config=None,
) -> np.ndarray:
    client = make_embedding_client(config)

    to_embed = texts if not cache else cache.get_missing_texts(texts)
    total = len(texts)
    remaining = len(to_embed)

    if log_path:
        update_log(log_path, f"[INFO] {remaining} of {total} items require new embeddings.")

    new_embeddings = []

    if to_embed:
        for i in tqdm(range(0, remaining, batch_size), desc="Computing embeddings"):
            batch = to_embed[i:i + batch_size]

            if log_path:
                update_log(
                    log_path,
                    f"[INFO] Embedding batch {i // batch_size + 1} ({len(batch)} items)...",
                )

            response = client.embeddings.create(
                model=model,
                input=batch
            )

            batch_embeddings = [
                np.array(d.embedding, dtype=np.float32)
                for d in response.data
            ]

            new_embeddings.extend(batch_embeddings)

            if cache:
                cache.add_embeddings(batch, np.vstack(batch_embeddings))
                if log_path:
                    update_log(log_path, f"[INFO] Cached {len(batch)} embeddings.")

    if cache:
        return cache.get_embeddings(texts)

    return np.vstack(new_embeddings)

def embed_texts(
    texts: List[str],
    model: str = "text-embedding-3-small",
    cache_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
    batch_size: int = 100,
    config=None,
) -> np.ndarray:
    """
    Create text embeddings using either:
    - OpenAI / OpenAI-compatible embeddings API
    - local sentence-transformers models, e.g. BAAI/bge-large-en-v1.5
    """
    if not texts:
        msg = "No texts provided for embedding."
        if log_path:
            update_log(log_path, f"[ERROR] {msg}")
        raise ValueError(msg)

    provider = _get_embedding_provider(config)
    model = _get_embedding_model(config, model)

    if config is not None:
        batch_size = getattr(config, "embedding_batch_size", batch_size)

    cache = EmbeddingCache(cache_path, model) if cache_path else None

    if log_path:
        update_log(
            log_path,
            f"[INFO] Starting embedding with provider='{provider}', model='{model}'. "
            f"Cache: {'enabled' if cache else 'disabled'}.",
        )

    if provider == "sentence_transformers":
        if cache:
            to_embed = cache.get_missing_texts(texts)
            if to_embed:
                embeddings = _embed_with_sentence_transformers(
                    to_embed,
                    model=model,
                    batch_size=batch_size,
                    log_path=log_path,
                )
                cache.add_embeddings(to_embed, embeddings)

            all_embeddings = cache.get_embeddings(texts)
        else:
            all_embeddings = _embed_with_sentence_transformers(
                texts,
                model=model,
                batch_size=batch_size,
                log_path=log_path,
            )

    elif provider in {"openai", "openai_compatible", "ollama"}:
        all_embeddings = _embed_with_openai_compatible(
            texts=texts,
            model=model,
            cache=cache,
            log_path=log_path,
            batch_size=batch_size,
            config=config,
        )

    else:
        raise ValueError(
            f"Unknown embedding_provider '{provider}'. "
            "Use 'openai', 'openai_compatible', 'ollama', or 'sentence_transformers'."
        )

    if log_path:
        update_log(log_path, f"[INFO] Embedding complete. Total items: {len(all_embeddings)}")

    return all_embeddings
