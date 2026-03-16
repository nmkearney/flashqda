# flashqda/embeddings/cache.py
"""
Embedding cache manager for FlashQDA.
Caches embeddings to disk to avoid recomputation and API costs.
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

class EmbeddingCache:
    """
    Handles storage and retrieval of embeddings on disk.
    Embeddings are stored as a dict: {text: embedding_list}.
    """

    def __init__(self, cache_path: Path, model_name: str):
        self.cache_path = Path(cache_path)
        self.model_name = model_name
        self.data: Dict[str, List[float]] = {}
        self._load_cache()

    # ---------------------------------------------------------------------
    def _load_cache(self):
        """Load cache from disk if it exists and matches the model name."""
        if not self.cache_path.exists():
            return

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            if raw.get("model") == self.model_name:
                self.data = raw.get("embeddings", {})
            else:
                print(f"⚠️ Cache model mismatch: expected {self.model_name}, found {raw.get('model')}. Ignoring old cache.")
        except Exception as e:
            print(f"⚠️ Failed to load embedding cache: {e}")

    # ---------------------------------------------------------------------
    def save_cache(self):
        """Save current cache to disk."""
        payload = {"model": self.model_name, "embeddings": self.data}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    # ---------------------------------------------------------------------
    def get_missing_texts(self, texts: List[str]) -> List[str]:
        """Return texts not already cached."""
        return [t for t in texts if t not in self.data]

    # ---------------------------------------------------------------------
    def add_embeddings(self, texts: List[str], embeddings: np.ndarray):
        """Add new embeddings to cache and persist to disk."""
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts and embeddings must match.")
        for text, emb in zip(texts, embeddings):
            self.data[text] = emb.tolist()
        self.save_cache()

    # ---------------------------------------------------------------------
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Retrieve cached embeddings for a list of texts.
        Returns a numpy array in the same order as input texts.
        """
        return np.array([self.data[t] for t in texts if t in self.data], dtype=np.float32)

    # ---------------------------------------------------------------------
    def has_all(self, texts: List[str]) -> bool:
        """Check if all given texts are cached."""
        return all(t in self.data for t in texts)

    # ---------------------------------------------------------------------
    def summary(self) -> str:
        """Return a short string summary of the cache."""
        return f"EmbeddingCache(model={self.model_name}, items={len(self.data)})"
