"""Tests for tools_filtering.py against TEST_FIXTURE.md known answers #5 and #6."""

import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_filtering import make_tools as make_filtering_tools
from tools_dtm import make_tools as make_dtm_tools


def test_pearson_and_spearman_filter_known_answer_6():
    """Validates pearson_filter_tool and spearman_filter_tool against known answer #6."""

    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM to populate session.feature_matrix and session.feature_names
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Get filtering tools
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Test pearson_filter_tool for target_class="catB"
    pearson_res = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})
    pearson_report = session.results_store[pearson_res["result_id"]]["full_report"]
    pearson_dict = dict(zip(pearson_report["term"], pearson_report["pearson_r"]))

    # Expected values from TEST_FIXTURE.md #6:
    # alpha: -0.6882, always: 0.3780, beta: -0.7746, delta: 1.0000, gamma: 0.8307
    assert pearson_dict["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert pearson_dict["always"] == pytest.approx(0.3780, abs=1e-3)
    assert pearson_dict["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert pearson_dict["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert pearson_dict["gamma"] == pytest.approx(0.8307, abs=1e-3)

    # Test spearman_filter_tool for target_class="catB"
    spearman_res = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})
    spearman_report = session.results_store[spearman_res["result_id"]]["full_report"]
    spearman_dict = dict(zip(spearman_report["term"], spearman_report["spearman_r"]))

    # Expected values from TEST_FIXTURE.md #6:
    # alpha: -0.7500, always: 0.3780, beta: -0.7746, delta: 1.0000, gamma: 0.9363
    assert spearman_dict["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert spearman_dict["always"] == pytest.approx(0.3780, abs=1e-3)
    assert spearman_dict["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert spearman_dict["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert spearman_dict["gamma"] == pytest.approx(0.9363, abs=1e-3)
