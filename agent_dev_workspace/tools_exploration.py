"""Exploration tools for pairwise document similarity and feature correlation analysis."""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from langchain_core.tools import tool


def make_tools(session):
    """Factory creating exploration tools bound to the given session state."""

    @tool
    def cosine_similarity_tool(doc_index_1: int, doc_index_2: int) -> Dict[str, Any]:
        """Calculates cosine similarity between two documents in session.feature_matrix using raw counts.

        Args:
            doc_index_1: Row index of the first document in the feature matrix.
            doc_index_2: Row index of the second document in the feature matrix.

        Returns:
            Dict containing document indices, calculated cosine similarity score, and result_id.
        """
        if session.feature_matrix is None:
            return {"status": "error", "message": "Document-term matrix has not been built yet."}

        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)
        n_docs = X.shape[0]

        if doc_index_1 < 0 or doc_index_1 >= n_docs:
            return {"status": "error", "message": f"doc_index_1 ({doc_index_1}) out of range [0, {n_docs-1}]."}

        if doc_index_2 < 0 or doc_index_2 >= n_docs:
            return {"status": "error", "message": f"doc_index_2 ({doc_index_2}) out of range [0, {n_docs-1}]."}

        v1 = X[doc_index_1, :].astype(float)
        v2 = X[doc_index_2, :].astype(float)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        # Handle zero vector edge case (e.g. empty text document)
        if norm1 == 0 or norm2 == 0:
            sim = 0.0
        else:
            sim = float(np.dot(v1, v2) / (norm1 * norm2))

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "similarity": round(sim, 6)
        }

        session.store_result("cosine_similarity_tool", {"doc_index_1": doc_index_1, "doc_index_2": doc_index_2}, summary)
        return summary

    return [cosine_similarity_tool]
