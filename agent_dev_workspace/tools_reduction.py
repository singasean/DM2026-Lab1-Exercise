import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from langchain_core.tools import tool
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

try:
    import umap
except ImportError:
    umap = None


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: str = "pca",
        n_components: int = 2,
        random_state: int = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
    ) -> dict:
        """Reduces feature matrix dimensions using PCA, TruncatedSVD, t-SNE, or UMAP, and generates a scatter plot.

        Args:
            method: The dimension reduction method to use ('pca', 'svd', 'tsne', or 'umap'). Default is 'pca'.
            n_components: Number of dimensions to reduce to. Default is 2.
            random_state: Seed for reproducible dimensionality reduction output. Default is 42.
            perplexity: Perplexity parameter for t-SNE reduction. Default is 30.0.
            n_neighbors: Number of neighbors parameter for UMAP reduction. Default is 15.

        Returns:
            dict: Summary containing result_id, reduced coordinates, method, and explained variance ratio if applicable.
        """
        if session.feature_matrix is None:
            return {"error": "No feature_matrix found in session. Please run build_dtm_tool first."}

        X = session.feature_matrix

        if method == "pca":
            if hasattr(X, "toarray"):
                X = X.toarray()
            model = PCA(n_components=n_components, random_state=random_state)
            coords = model.fit_transform(X)
            explained_variance = model.explained_variance_ratio_.tolist()
        elif method in ("svd", "truncated_svd"):
            model = TruncatedSVD(n_components=n_components, random_state=random_state)
            coords = model.fit_transform(X)
            explained_variance = model.explained_variance_ratio_.tolist()
        elif method == "tsne":
            if hasattr(X, "toarray"):
                X = X.toarray()
            model = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
            coords = model.fit_transform(X)
            explained_variance = None
        elif method == "umap":
            if umap is None:
                return {"error": "umap-learn library is not installed."}
            if hasattr(X, "toarray"):
                X = X.toarray()
            model = umap.UMAP(n_components=n_components, random_state=random_state, n_neighbors=n_neighbors)
            coords = model.fit_transform(X)
            explained_variance = None
        else:
            return {"error": f"Unsupported method '{method}'. Choose from 'pca', 'svd', 'tsne', or 'umap'."}

        coords_list = coords.tolist()
        session.artifacts["reduced_coordinates"] = coords_list

        # Plotting requirement: scatter plot colored by category
        fig, ax = plt.subplots(figsize=(8, 6))
        categories = None
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            categories = session.dataframe["category_name"].values
        elif session.dataframe is not None and "category" in session.dataframe.columns:
            categories = session.dataframe["category"].values
        elif session.labels is not None:
            categories = session.labels

        if categories is not None and len(categories) == coords.shape[0]:
            sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=categories, palette="tab10", ax=ax)
        else:
            sns.scatterplot(x=coords[:, 0], y=coords[:, 1], ax=ax)

        ax.set_title(f"Dimension Reduction ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "n_samples": coords.shape[0],
            "coordinates": coords_list,
        }
        if explained_variance is not None:
            summary["explained_variance_ratio"] = explained_variance

        session.store_result("reduce_dimensions_tool", {"method": method, "n_components": n_components}, summary)

        return summary

    return [reduce_dimensions_tool]
