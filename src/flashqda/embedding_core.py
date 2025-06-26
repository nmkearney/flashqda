# embedding_core.py
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def _concept_key(concept_type, text):
    return f"{concept_type}::{text.strip()}"

def compute_embedding(text, model=None):
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model.encode(text, convert_to_numpy=True).tolist()
