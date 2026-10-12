"""Tests for tools_patterns.py against TEST_FIXTURE.md known answers #17 and #20."""

import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


def test_mine_patterns_known_answers_17_and_20():
    """Validates mine_patterns_tool against known answers #17 and #20."""

    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # --- Known Answer #17: fpgrowth with tfidf ---
    # catA, tfidf, fpgrowth, min_sup=1 -> pattern ["alpha"], support 3
    catA_res = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert len(catA_res["patterns"]) == 1
    assert catA_res["patterns"][0]["pattern"] == ["alpha"]
    assert catA_res["patterns"][0]["support"] == 3

    # catB, tfidf, fpgrowth, min_sup=1 -> pattern ["gamma"], support 4
    catB_res = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert len(catB_res["patterns"]) == 1
    assert catB_res["patterns"][0]["pattern"] == ["gamma"]
    assert catB_res["patterns"][0]["support"] == 4

    # Zero terms survived check: variance / term_frequency filtering keeps zero terms on sample fixture
    var_res = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert var_res["patterns"] == []
    assert "no terms survived" in var_res.get("note", "").lower()

    # --- Known Answer #20: topk and maxfpgrowth ---
    # topk with k=1: catA -> ["alpha"] support 3, catB -> ["gamma"] support 4
    topk_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert len(topk_catA["patterns"]) == 1
    assert topk_catA["patterns"][0]["pattern"] == ["alpha"]
    assert topk_catA["patterns"][0]["support"] == 3

    # maxfpgrowth with min_sup=1: returns zero patterns for both categories
    max_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert max_catA["patterns"] == []
