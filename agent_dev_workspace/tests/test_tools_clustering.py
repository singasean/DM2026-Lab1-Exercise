import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_clustering import make_tools


def test_kmeans_tool_fixture():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(session.dataframe["text"].fillna(""))
    session.feature_matrix = X
    session.feature_names = vectorizer.get_feature_names_out().tolist()

    tools = make_tools(session)
    kmeans_tool = next(t for t in tools if t.name == "kmeans_tool")

    result = kmeans_tool.invoke({"n_clusters": 2, "random_state": 42})

    assert "result_id" in result
    assert result["n_clusters"] == 2
    assert len(result["cluster_labels"]) == 8
    assert "counts_per_cluster" in result
    assert "top_terms_per_cluster" in result
    assert session.artifacts["cluster_labels"] == result["cluster_labels"]
