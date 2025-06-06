import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def _concept_key(concept_type, text):
    return f"{concept_type}::{text.strip()}"

def load_embeddings(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_embeddings(embeddings, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, indent=2)

def compute_embedding(text):
    return model.encode(text, convert_to_numpy=True).tolist()

def update_embeddings_from_data(data_list, embeddings_path):
    existing = load_embeddings(embeddings_path)
    updated = dict(existing)  # Copy to modify
    changed_keys = []

    concepts_to_embed = []

    for entry in data_list:
        for key_type in ["cause", "effect"]:
            text = entry.get(key_type, "").strip()
            if not text:
                continue
            key = _concept_key(key_type, text)

            if key not in existing or existing[key]["text"] != text:
                concepts_to_embed.append((key_type, text, key))

    for key_type, text, key in tqdm(concepts_to_embed, desc="🔄 Generating embeddings"):
        embedding = compute_embedding(text)
        updated[key] = {"text": text, "embedding": embedding}
        changed_keys.append(key)

    if changed_keys:
        save_embeddings(updated, embeddings_path)

    return updated, changed_keys

def cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def compute_similarity_matrix(embeddings):
    causes = {k: v for k, v in embeddings.items() if k.startswith("cause::")}
    effects = {k: v for k, v in embeddings.items() if k.startswith("effect::")}
    matrix = {}

    for c_key, c_val in causes.items():
        matrix[c_key] = {}
        for e_key, e_val in effects.items():
            score = cosine_similarity(c_val["embedding"], e_val["embedding"])
            matrix[c_key][e_key] = score

    return matrix
