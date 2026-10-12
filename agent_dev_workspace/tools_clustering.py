from typing import List
import numpy as np
from langchain_core.tools import tool
from sklearn.cluster import KMeans


def make_tools(session):
    @tool
    def kmeans_tool(
        n_clusters: int = 3,
        random_state: int = 42,
        n_init: int = 10,
    ) -> dict:
        """Clusters documents in the feature matrix using K-Means clustering.

        Args:
            n_clusters: The number of clusters to form. Default is 3.
            random_state: Seed for reproducible cluster centroid initialization. Default is 42.
            n_init: Number of times the k-means algorithm will be run with different centroid seeds. Default is 10.

        Returns:
            dict: Summary containing result_id, cluster_labels, counts_per_cluster, and top terms per cluster if feature names exist.
        """
        if session.feature_matrix is None:
            return {"error": "No feature_matrix found in session. Please run build_dtm_tool first."}

        X = session.feature_matrix

        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(X).tolist()

        session.artifacts["cluster_labels"] = labels

        # Compute count per cluster
        unique, counts = np.unique(labels, return_counts=True)
        counts_per_cluster = {f"cluster_{u}": int(c) for u, c in zip(unique, counts)}

        # Find top terms per cluster center if feature_names exist
        top_terms_per_cluster = {}
        if session.feature_names is not None and len(session.feature_names) == model.cluster_centers_.shape[1]:
            feature_names = np.array(session.feature_names)
            for cluster_idx, center in enumerate(model.cluster_centers_):
                top_indices = center.argsort()[::-1][:5]
                top_terms_per_cluster[f"cluster_{cluster_idx}"] = feature_names[top_indices].tolist()

        result_id = session.next_result_id("kmeans")
        summary = {
            "result_id": result_id,
            "n_clusters": n_clusters,
            "counts_per_cluster": counts_per_cluster,
            "cluster_labels": labels,
        }
        if top_terms_per_cluster:
            summary["top_terms_per_cluster"] = top_terms_per_cluster

        session.store_result("kmeans_tool", {"n_clusters": n_clusters, "random_state": random_state}, summary)

        return summary

    return [kmeans_tool]
