# clusterer.py

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster


def hierarchical_cluster_items(
    texts: List[str],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.8,
    linkage_method: str = "average",
) -> Tuple[Dict[int, List[str]], pd.DataFrame, np.ndarray]:
    """
    Group semantically similar items using Agglomerative Hierarchical Clustering (AHC).

    Args:
        texts:
            List of unique strings we are clustering. Index i must align with embeddings[i].
        embeddings:
            2D array of shape (n_items, dim).
        similarity_threshold:
            Minimum within-cluster cosine similarity we accept.
            We'll convert this to a distance threshold: distance = 1 - similarity.
            Example: 0.8 similarity -> 0.2 distance.
        linkage_method:
            'average' is typical for cosine; 'complete' is stricter.

    Returns:
        clusters:
            {cluster_id: [item_text, ...]} where each item appears exactly once.
        sim_df:
            pandas DataFrame of cosine similarities (n x n), index/cols = item texts.
        Z:
            scipy linkage matrix. Use for dendrogram plotting/audit.
    """
    # 1. Cosine similarity matrix
    sim_matrix = cosine_similarity(embeddings)  # shape (n,n)

    # 2. Convert to distance matrix for clustering
    #    cosine distance = 1 - cosine similarity
    distance_matrix = 1.0 - sim_matrix

    # 3. Condense the distance matrix to the upper triangle (scipy linkage API)
    #    linkage() expects condensed distance form, like scipy.spatial.distance.pdist.
    tri_upper = np.triu_indices(len(distance_matrix), k=1)
    condensed_distances = distance_matrix[tri_upper]

    # 4. Run hierarchical clustering
    #    average linkage on cosine distance is standard for semantic text clustering.
    Z = linkage(condensed_distances, method=linkage_method)

    # 5. Cut the dendrogram at the chosen distance threshold
    #    If similarity_threshold = 0.8, distance_threshold = 0.2
    distance_threshold = 1.0 - similarity_threshold
    cluster_assignments = fcluster(
        Z,
        t=distance_threshold,
        criterion="distance",
    )

    # 6. Build cluster mapping
    clusters: Dict[int, List[str]] = {}
    for item_text, cid in zip(texts, cluster_assignments):
        clusters.setdefault(cid, []).append(item_text)

    # 7. Cosine similarity DataFrame (for audit/export)
    sim_df = pd.DataFrame(sim_matrix, index=texts, columns=texts)

    return clusters, sim_df, Z
