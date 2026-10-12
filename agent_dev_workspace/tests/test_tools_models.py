import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_models import make_tools


def test_train_test_split_tool_fixture():
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
    split_tool = next(t for t in tools if t.name == "train_test_split_tool")

    result = split_tool.invoke({"test_size": 0.25, "random_state": 42})

    assert "result_id" in result
    assert result["n_train_samples"] == 6
    assert result["n_test_samples"] == 2
    assert result["n_features"] == 5

    assert "train_test_split" in session.artifacts
    artifacts = session.artifacts["train_test_split"]
    assert artifacts["X_train"].shape[0] == 6
    assert artifacts["X_test"].shape[0] == 2


def test_naive_bayes_tool_fixture():
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
    naive_bayes_tool = next(t for t in tools if t.name == "naive_bayes_tool")

    result = naive_bayes_tool.invoke({"alpha": 1.0, "use_split": False})

    assert "result_id" in result
    assert result["model"] == "MultinomialNB"
    assert result["accuracy"] == 1.0
    assert result["n_eval_samples"] == 8
    assert "predictions" in session.artifacts
    assert len(session.artifacts["predictions"]) == 8
