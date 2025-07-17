# embedding_cache.py
import os
import json
import tempfile
from tqdm import tqdm
from .embedding_core import _concept_key, compute_embedding

def load_embeddings(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_embeddings(embeddings, path):
    temp_path = str(path) + ".tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, indent=2)
    os.replace(temp_path, path)

def update_embeddings_from_data(data_list, embeddings_path, concept_types=["cause", "effect"], model=None):
    existing = load_embeddings(embeddings_path)
    updated = dict(existing)  # Copy to modify
    changed_keys = []

    concepts_to_embed = []

    for entry in data_list:
        for key_type in concept_types:
            text = entry.get(key_type, "").strip()
            if not text:
                continue
            key = _concept_key(key_type, text)

            if key not in existing or existing[key]["text"] != text:
                concepts_to_embed.append((key_type, text, key))

    for key_type, text, key in tqdm(concepts_to_embed, desc="🔄 Generating embeddings"):
        embedding = compute_embedding(text, model)
        updated[key] = {"text": text, "embedding": embedding}
        changed_keys.append(key)

    if changed_keys:
        save_embeddings(updated, embeddings_path)

    return updated, changed_keys