# embedding_analysis.py
from .embedding_core import cosine_similarity

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