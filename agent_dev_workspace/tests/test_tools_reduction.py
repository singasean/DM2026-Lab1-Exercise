import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools


def test_pca_reduction_known_answer():
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
    reduce_dimensions_tool = next(t for t in tools if t.name == "reduce_dimensions_tool")

    result = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2, "random_state": 42})

    expected_coords = np.array(
        [
            [2.3669, 0.8871],
            [1.7039, 0.2642],
            [1.0408, -0.3587],
            [-2.0759, 1.0129],
            [-1.4554, 0.3405],
            [-0.8349, -0.3319],
            [-0.8349, -0.3319],
            [0.0894, -1.4822],
        ]
    )

    expected_variance = [0.7532, 0.1965]

    coords = np.array(result["coordinates"])
    explained_variance = np.array(result["explained_variance_ratio"])

    np.testing.assert_allclose(coords, expected_coords, atol=1e-3)
    np.testing.assert_allclose(explained_variance, expected_variance, atol=1e-3)
    assert session.pending_figure is not None


def test_svd_reduction_fixture():
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
    reduce_dimensions_tool = next(t for t in tools if t.name == "reduce_dimensions_tool")

    result = reduce_dimensions_tool.invoke({"method": "svd", "n_components": 2, "random_state": 42})

    assert result["method"] == "svd"
    assert len(result["coordinates"]) == 8
    assert len(result["coordinates"][0]) == 2
    assert "explained_variance_ratio" in result
    assert session.pending_figure is not None
