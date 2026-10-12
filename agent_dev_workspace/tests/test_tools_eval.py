import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_eval import make_tools


def test_evaluate_model_tool_fixture():
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

    y_true = session.dataframe["category"].values
    # Perfect predictions matching ground truth
    session.artifacts["predictions"] = y_true.tolist()

    tools = make_tools(session)
    eval_tool = next(t for t in tools if t.name == "evaluate_model_tool")

    result = eval_tool.invoke({"average": "macro"})

    assert "result_id" in result
    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1_score"] == 1.0
    assert result["n_samples"] == 8
    assert result["confusion_matrix"] == [[4, 0], [0, 4]]
