"""
Unit tests for agent_1_classifier.classify_paper.

All tests run fully offline — no real Anthropic API calls are made.
The Anthropic client is replaced with a MagicMock whose
``messages.create`` method returns a pre-built fake response.

Run with:
    pytest tests/test_classifier.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — mirrors the pattern used in test_parser.py.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.agent_1_classifier import classify_paper  # noqa: E402
from schemas.parsed_paper import ParsedPaper  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture: load the shared mock paper
# ---------------------------------------------------------------------------

_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "mock_paper.json"


@pytest.fixture()
def mock_paper() -> ParsedPaper:
    """Return a ParsedPaper loaded from tests/fixtures/mock_paper.json."""
    return ParsedPaper.load(_FIXTURE_PATH)


# ---------------------------------------------------------------------------
# Helper: build a fake Anthropic response object
# ---------------------------------------------------------------------------


def _make_mock_response(json_str: str) -> MagicMock:
    """Create a MagicMock that looks like an anthropic.Message response.

    Args:
        json_str: JSON string the mock LLM should 'return'.

    Returns:
        A MagicMock with ``.content[0].text`` and ``.usage`` attributes set.
    """
    resp = MagicMock()
    resp.content = [MagicMock(text=json_str)]
    resp.usage.input_tokens = 100
    resp.usage.output_tokens = 50
    return resp


# ---------------------------------------------------------------------------
# Shared JSON templates for mock LLM responses
# ---------------------------------------------------------------------------

def _eqr_json(paper_id: str = "mock01") -> str:
    return json.dumps({
        "Paper_ID": paper_id,
        "Classification_Code": "EQR",
        "Eligible_Extraction": "Yes",
        "Confidence_Score": 4,
        "Classification_Basis": (
            "The paper uses OLS regression with firm and year fixed effects "
            "to test hypotheses about board diversity and firm risk."
        ),
        "Flag": "",
        "Requires_Human_Review": False,
    })


def _ct_json(paper_id: str = "mock01") -> str:
    return json.dumps({
        "Paper_ID": paper_id,
        "Classification_Code": "CT",
        "Eligible_Extraction": "No",
        "Confidence_Score": 3,
        "Classification_Basis": (
            "The paper presents a conceptual framework without original "
            "hypothesis testing or regression analysis."
        ),
        "Flag": "",
        "Requires_Human_Review": False,
    })


def _mm_json(paper_id: str = "mock01") -> str:
    return json.dumps({
        "Paper_ID": paper_id,
        "Classification_Code": "MM",
        "Eligible_Extraction": "Conditional",
        "Confidence_Score": 3,
        "Classification_Basis": (
            "The paper combines semi-structured interviews with a regression "
            "analysis of survey data."
        ),
        "Flag": "",
        "Requires_Human_Review": False,
    })


# ---------------------------------------------------------------------------
# Test 1 — EQR stable paper
# ---------------------------------------------------------------------------


def test_eqr_stable(mock_paper: ParsedPaper, tmp_path: Path) -> None:
    """Both classifier runs agree on EQR — result should be stable with no flags."""
    client = MagicMock()
    client.messages.create.return_value = _make_mock_response(_eqr_json())

    result = classify_paper(mock_paper, client, output_dir=tmp_path)

    assert result["Classification_Code"] == "EQR"
    assert result["Eligible_Extraction"] == "Yes"   # from ELIGIBLE_MAP, overrides LLM
    assert result["Confidence_Score"] == 4
    assert result["Requires_Human_Review"] is False
    assert result["run1_code"] == "EQR"
    assert result["run2_code"] == "EQR"
    assert result["Flag"] == ""


# ---------------------------------------------------------------------------
# Test 2 — CT paper with empty methods section
# ---------------------------------------------------------------------------


def test_ct_empty_methods(tmp_path: Path) -> None:
    """Empty methods_data triggers FLAG-INCOMPLETE-SOURCE and forces human review."""
    # Build a paper with blank methods_data.
    paper = ParsedPaper(
        paper_id="mock_ct",
        title="A Conceptual Review of Corporate Governance Theories",
        authors="Smith, A.",
        abstract=(
            "This paper reviews competing theoretical perspectives on corporate "
            "governance, synthesising agency theory, stakeholder theory, and "
            "resource dependence theory."
        ),
        theory_hypotheses="",
        methods_data="",          # <-- intentionally empty
        results="",
        discussion="",
        footnotes="",
        appendix="",
        full_text="A Conceptual Review of Corporate Governance Theories\nSmith, A.",
        page_count=18,
        parse_quality="medium",
        parse_notes="Extracted via pdfplumber. Missing or short core sections: ['methods_data'].",
    )

    client = MagicMock()
    client.messages.create.return_value = _make_mock_response(_ct_json("mock_ct"))

    result = classify_paper(paper, client, output_dir=tmp_path)

    assert result["Classification_Code"] == "CT"
    assert result["Eligible_Extraction"] == "No"
    assert "[FLAG-INCOMPLETE-SOURCE]" in result["Flag"]
    assert result["Requires_Human_Review"] is True


# ---------------------------------------------------------------------------
# Test 3 — MM edge case: unstable runs trigger FLAG-CLASSIFICATION-UNSTABLE
# ---------------------------------------------------------------------------


def test_mm_unstable_runs(tmp_path: Path) -> None:
    """When run1=MM and run2=EQR differ, pipeline flags the result as unstable."""
    paper = ParsedPaper(
        paper_id="mock_mm",
        title="Leadership Styles and Innovation: A Mixed-Methods Investigation",
        authors="Patel, R. and Nguyen, T.",
        abstract=(
            "We combine semi-structured interviews with senior managers and a "
            "survey-based regression analysis to examine how transformational "
            "leadership influences firm-level innovation."
        ),
        theory_hypotheses="H1: Transformational leadership is positively associated with innovation.",
        methods_data=(
            "Phase 1: We conducted 24 semi-structured interviews with C-suite executives "
            "across manufacturing firms. Interviews were coded using NVivo following a "
            "grounded theory approach. Phase 2: We distributed a structured survey to "
            "412 firms and estimated OLS regressions with industry and region fixed effects "
            "to test H1. The dependent variable is a composite innovation index."
        ),
        results="OLS results (beta=0.31, p<0.01) support H1. Qualitative themes reinforce the finding.",
        discussion="Both methods converge on the conclusion that transformational leadership drives innovation.",
        footnotes="",
        appendix="",
        full_text="Leadership Styles and Innovation...",
        page_count=28,
        parse_quality="high",
        parse_notes="",
    )

    mm_resp = _make_mock_response(_mm_json("mock_mm"))
    eqr_resp = _make_mock_response(_eqr_json("mock_mm"))

    client = MagicMock()
    # run1 returns MM, run2 returns EQR — codes differ → unstable.
    client.messages.create.side_effect = [mm_resp, eqr_resp]

    result = classify_paper(paper, client, output_dir=tmp_path)

    assert result["Classification_Code"] == "MM"          # run1 wins
    assert result["Eligible_Extraction"] == "Conditional"
    assert "[FLAG-CLASSIFICATION-UNSTABLE]" in result["Flag"]
    assert result["Requires_Human_Review"] is True
