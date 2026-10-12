from typing import List, Optional, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session: Any) -> List[Any]:
    """Factory function creating tools bound to the provided session instance."""

    @tool
    def build_dtm_tool(
        text_column: str = "text",
        ngram_range: List[int] = [1, 1],
        min_df: float = 1,
        max_df: float = 1.0,
        max_features: Optional[int] = None,
        stop_words: Optional[str] = None
    ) -> Dict[str, Any]:
        """Builds a Document-Term Matrix (DTM) from the dataset text column.

        Extracts unigram or n-gram counts across all documents using CountVectorizer,
        populates session.feature_matrix and session.feature_names, calculates matrix sparsity,
        and saves the fitted vectorizer artifact for downstream tools.

        Args:
            text_column: Name of the text column in session.dataframe. Default is "text".
            ngram_range: Lower and upper boundary of range of n-grams as a list of 2 ints, e.g. [1, 1].
            min_df: Minimum document frequency threshold (float for fraction, int for absolute count).
            max_df: Maximum document frequency threshold (float for fraction, int for absolute count).
            max_features: Maximum number of vocabulary features to build, or None for all.
            stop_words: Language for stop words removal (e.g., "english"), or None.

        Returns:
            Dict summarizing the DTM construction (result_id, n_documents, n_terms, sparsity_pct, etc.).
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session.dataframe. Call load_dataset_tool first."}

        if text_column not in session.dataframe.columns:
            return {"error": f"Column '{text_column}' not found in session.dataframe."}

        # Convert list to tuple for scikit-learn CountVectorizer
        ngram_tuple = (ngram_range[0], ngram_range[1]) if len(ngram_range) >= 2 else (1, 1)

        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            stop_words=stop_words
        )

        texts = session.dataframe[text_column].fillna("")
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = list(vectorizer.get_feature_names_out())

        # Update session state
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = float(100.0 * (1.0 - (non_zero / total_elements))) if total_elements > 0 else 0.0

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": int(n_rows),
            "n_terms": int(n_cols),
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "feature_names": feature_names
        }

        session.store_result("build_dtm_tool", {"text_column": text_column, "ngram_range": ngram_range}, summary)
        return summary

    return [build_dtm_tool]
