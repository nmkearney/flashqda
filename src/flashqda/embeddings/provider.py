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
from flashqda.llm_utils import get_client_kwargs, safe_llm_call
from openai import OpenAI, OpenAIError
from flashqda.log_utils import update_log


def embed_texts(
    texts: List[str],
    model: str = "text-embedding-3-small",
    cache_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
    batch_size: int = 100,
    config=None, 
) -> np.ndarray:
    """
    Create text embeddings using LLM with FlashQDA-safe conventions.

    - Uses centralized client and retry-safe API wrapper.
    - Supports checkpoint-like progress tracking via cache.
    - Logs all progress and warnings to project log.

    Args:
        texts: List of text strings to embed.
        model: Embedding model name.
        cache_path: Path to JSON cache file for embeddings.
        log_path: Optional log file path for structured logging.
        batch_size: Number of texts per API batch.
        
    Returns:
        np.ndarray of shape (n_texts, embedding_dim)
    """
    if not texts:
        msg = "No texts provided for embedding."
        if log_path:
            update_log(log_path, f"[ERROR] {msg}")
        raise ValueError(msg)

    # Initialize API client
    client = OpenAI(**get_client_kwargs(config))

    cache = EmbeddingCache(cache_path, model) if cache_path else None

    if log_path:
        update_log(log_path, f"Starting embedding with model '{model}'. "
                             f"Cache: {'enabled' if cache else 'disabled'}.")

    # Determine which items are missing
    to_embed = texts if not cache else cache.get_missing_texts(texts)
    total = len(texts)
    remaining = len(to_embed)

    if log_path:
        update_log(log_path, f"{remaining} of {total} items require new embeddings.")

    # Generate new embeddings in batches
    new_embeddings = []
    if to_embed:
        for i in tqdm(range(0, remaining, batch_size), desc="Computing embeddings"):
        # for i in range(0, remaining, batch_size):
            batch = to_embed[i:i + batch_size]
            if log_path:
                update_log(log_path, f"Embedding batch {i // batch_size + 1} "
                                     f"({len(batch)} items)...")

            # Safe LLM call
            response = safe_llm_call(
                client.embeddings.create,
                model=model,
                input=batch
                )

            batch_embeddings = [np.array(d.embedding, dtype=np.float32) for d in response.data]
            new_embeddings.extend(batch_embeddings)

            # Save partial results to cache (checkpoint behavior)
            if cache:
                cache.add_embeddings(batch, np.vstack(batch_embeddings))
                if log_path:
                    update_log(log_path, f"Cached {len(batch)} embeddings.")

    # Combine cached and new embeddings
    if cache:
        all_embeddings = cache.get_embeddings(texts)
        if log_path:
            update_log(log_path, f"Loaded {len(all_embeddings)} total embeddings from cache.")
    else:
        all_embeddings = np.vstack(new_embeddings)

    if log_path:
        update_log(log_path, f"Embedding complete. Total items: {len(all_embeddings)}")

    return all_embeddings
