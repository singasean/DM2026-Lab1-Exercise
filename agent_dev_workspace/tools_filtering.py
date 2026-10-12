"""Filtering tools for feature selection based on variance and correlation metrics."""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
from langchain_core.tools import tool


def make_tools(session):
    """Factory creating feature filtering tools bound to the given session state."""

    @tool
    def variance_filter_tool(threshold: float = 0.0) -> Dict[str, Any]:
        """Flags low-variance terms in the Document-Term Matrix for feature selection.

        Calculates variance across documents for each term in session.feature_matrix.
        Terms with variance below the threshold are flagged for removal. Stores a DataFrame
        report suitable for visualize_result_tool.

        Args:
            threshold: Minimum variance threshold required to keep a feature. Default is 0.0.

        Returns:
            Dict summarizing total, kept, and removed features along with variance details.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No Document-Term Matrix in session. Build DTM using build_dtm_tool first."}

        # Convert sparse matrix to dense array for variance calculation across rows (axis=0)
        dense_matrix = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)
        variances = np.var(dense_matrix, axis=0)

        feature_names = session.feature_names
        report_data = []
        kept_features = []
        removed_features = []

        for name, var in zip(feature_names, variances):
            var_val = float(var)
            report_data.append({"term": str(name), "variance": var_val})
            if var_val >= threshold:
                kept_features.append(str(name))
            else:
                removed_features.append(str(name))

        # Create full_report DataFrame sorted descending by variance
        report_df = pd.DataFrame(report_data)
        report_df.sort_values(by="variance", ascending=False, inplace=True)
        report_df.reset_index(drop=True, inplace=True)

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "status": "success",
            "threshold": threshold,
            "total_features": len(feature_names),
            "kept_features_count": len(kept_features),
            "removed_features_count": len(removed_features),
            "kept_features": kept_features,
            "removed_features": removed_features,
            "variances": {row["term"]: round(row["variance"], 6) for row in report_data}
        }

        # Store result with full_report for visualization tools
        session.store_result("variance_filter_tool", {"threshold": threshold}, summary, full_report=report_df)
        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> Dict[str, Any]:
        """Calculates Pearson correlation between term frequencies and a target class indicator.

        Args:
            target_class: The category name to correlate against (1 for documents in this category, 0 otherwise).
            threshold: Minimum absolute Pearson correlation value required to keep a term. Defaults to 0.0.

        Returns:
            Dict containing selected features, total evaluated, and the result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "Document-term matrix has not been built yet."}

        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values

        if labels is None:
            return {"status": "error", "message": "No category labels found in session."}

        y = (np.array(labels) == target_class).astype(float)
        if y.sum() == 0:
            return {"status": "error", "message": f"Target class '{target_class}' not found in labels."}

        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else session.feature_matrix
        n_features = X.shape[1]

        correlations = []
        for i in range(n_features):
            x = X[:, i]
            if np.std(x) == 0 or np.std(y) == 0:
                r = 0.0
            else:
                r, _ = stats.pearsonr(x, y)
                if np.isnan(r):
                    r = 0.0
            correlations.append(r)

        report_df = pd.DataFrame({
            "term": list(session.feature_names),
            "pearson_r": correlations
        })

        report_df = report_df.sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        selected_df = report_df[report_df["pearson_r"].abs() >= threshold]
        selected_terms = selected_df["term"].tolist()

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_features": int(n_features),
            "selected_features_count": len(selected_terms),
            "selected_terms": selected_terms
        }

        session.store_result("pearson_filter_tool", {"target_class": target_class, "threshold": threshold}, summary, full_report=report_df)
        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0) -> Dict[str, Any]:
        """Calculates Spearman rank correlation between term frequencies and a target class indicator.

        Args:
            target_class: The category name to correlate against (1 for documents in this category, 0 otherwise).
            threshold: Minimum absolute Spearman correlation value required to keep a term. Defaults to 0.0.

        Returns:
            Dict containing selected features, total evaluated, and the result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "Document-term matrix has not been built yet."}

        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values

        if labels is None:
            return {"status": "error", "message": "No category labels found in session."}

        y = (np.array(labels) == target_class).astype(float)
        if y.sum() == 0:
            return {"status": "error", "message": f"Target class '{target_class}' not found in labels."}

        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else session.feature_matrix
        n_features = X.shape[1]

        correlations = []
        for i in range(n_features):
            x = X[:, i]
            if np.std(x) == 0 or np.std(y) == 0:
                rho = 0.0
            else:
                rho, _ = stats.spearmanr(x, y)
                if np.isnan(rho):
                    rho = 0.0
            correlations.append(rho)

        report_df = pd.DataFrame({
            "term": list(session.feature_names),
            "spearman_r": correlations
        })

        report_df = report_df.sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        selected_df = report_df[report_df["spearman_r"].abs() >= threshold]
        selected_terms = selected_df["term"].tolist()

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_features": int(n_features),
            "selected_features_count": len(selected_terms),
            "selected_terms": selected_terms
        }

        session.store_result("spearman_filter_tool", {"target_class": target_class, "threshold": threshold}, summary, full_report=report_df)
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
