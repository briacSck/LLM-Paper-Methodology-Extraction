"""
Unit tests for schemas.extraction_schema.ExtractionResult.

No LLM calls are made — all tests exercise the Pydantic schema layer directly.

Run with:
    pytest tests/test_extractor.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from schemas.extraction_schema import ExtractionResult, EXCEL_FIELDS  # noqa: E402
from schemas.parsed_paper import ParsedPaper  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "mock_paper.json"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _valid_extraction_dict(**overrides) -> dict:
    """Return a fully-populated extraction dict with plausible values.

    All 65 Excel fields are explicitly set so ``to_excel_dict()`` contains no
    None values.  Any field can be overridden via keyword arguments.
    """
    base = {
        # Cat 1 — Bibliographic
        "Paper_ID": "mock01",
        "Authors": "Harrington, J., Chen, L., and Osei, K.",
        "Year": "2023",
        "Journal": "Strategic Management Journal",
        "Title": "Board Diversity and Firm Risk: A Panel Study of S&P 500 Companies",
        # Cat 2 — Research design
        "RQ_Summary": "Does board gender diversity reduce total firm risk in S&P 500 firms?",
        "Num_Hypotheses": "2",
        "Primary_Relationship": "Board diversity → firm risk",
        "Relationship_Direction": "Negative",
        # Cat 3 — DV
        "DV_Name": "Total firm risk",
        "DV_Construct": "Annualised return volatility",
        "DV_Measurement": "Standard deviation of daily stock returns, annualised",
        "DV_Source": "Compustat / CRSP",
        "DV_Type": "1",
        "DV_Num": "1",
        # Cat 4 — IV
        "IV_Name": "Board gender diversity",
        "IV_Construct": "Proportion of female board members",
        "IV_Measurement": "Female directors / total directors",
        "IV_Source": "ISS Governance database",
        "IV_Type": "1",
        "IV_Num": "1",
        # Cat 5 — Mediation
        "Mediator_Present": "0",
        "Mediator_Name": "NA",
        "Mediator_Construct": "NA",
        "Mediator_Measurement": "NA",
        "Mediation_Method": "NA",
        # Cat 6 — Moderation
        "Moderator_Present": "0",
        "Moderator_Name": "NA",
        "Moderator_Construct": "NA",
        "Moderator_Measurement": "NA",
        "Moderation_Method": "NA",
        # Cat 7 — Controls
        "Control_Num": "4",
        "Control_List": "Firm size, leverage, profitability, board size",
        "Control_Justified": "0",
        # Cat 8 — Sample / Data
        "Sample_Size": "4750",
        "Sample_Context": "S&P 500 firms, 2010-2020",
        "Data_Type": "2",
        "Data_Source_Primary": "Compustat; ISS Governance",
        "Unit_of_Analysis": "Firm-year",
        "Time_Period": "2010-2020",
        # Cat 9 — Model / Estimation
        "Model_Type": "3",
        "Model_Type_Other": "NR",
        "Endogeneity_Addressed": "0",
        "Endogeneity_Method": "NA",
        "Robustness_Checks": "1",
        # Cat 10 — Missing Data
        "Missing_Mentioned": "1",
        "Missing_Rate_Reported": "0",
        "Missing_Rate_Value": "NR",
        "Missing_Variables": "NR",
        "Missing_Handling": "1",
        "Missing_Handling_Other": "NR",
        "Missing_Justified": "0",
        "Missing_Pattern_Tested": "0",
        "Missing_Pattern_Result": "NR",
        "Missing_Sensitivity": "0",
        # Cat 11 — Transparency
        "Data_Available": "3",
        "Code_Available": "0",
        "Software_Reported": "1",
        "Software_Name": "Stata 17",
        "Replication_Feasibility": "2",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Test 1 — Valid extraction passes Pydantic validation
# ---------------------------------------------------------------------------


def test_valid_extraction_roundtrip() -> None:
    """A fully-populated ExtractionResult validates and serialises correctly."""
    # Confirm the fixture file is parseable (validates ParsedPaper schema).
    ParsedPaper.load(_FIXTURE_PATH)

    result = ExtractionResult(**_valid_extraction_dict())

    assert result.Paper_ID == "mock01"
    assert result.Model_Type == "3"
    assert result.Missing_Handling == "1"

    excel = result.to_excel_dict()

    # Must contain exactly the 65 Excel columns.
    assert set(excel.keys()) == set(EXCEL_FIELDS)

    # No None values — the _coerce_and_warn validator converts None → "NR".
    none_fields = [k for k, v in excel.items() if v is None]
    assert none_fields == [], f"Unexpected None values in to_excel_dict(): {none_fields}"


# ---------------------------------------------------------------------------
# Test 2 — Wrong type raises Pydantic ValidationError
# ---------------------------------------------------------------------------


def test_invalid_confidence_type_raises() -> None:
    """Passing a non-numeric string for Extraction_Confidence must raise ValidationError."""
    with pytest.raises(ValidationError):
        ExtractionResult(
            **_valid_extraction_dict(),
            Extraction_Confidence="banana",   # str → int coercion fails
        )


# ---------------------------------------------------------------------------
# Test 3 — FLAG-QC-MISSING-LOGIC triggered by inconsistent missing-data fields
# ---------------------------------------------------------------------------


def test_missing_logic_flag_and_correction() -> None:
    """Missing_Mentioned='0' with Missing_Handling='1' triggers FLAG-QC-MISSING-LOGIC
    and corrects Missing_Handling to 'NA'."""
    result = ExtractionResult(
        **_valid_extraction_dict(
            Missing_Mentioned="0",
            Missing_Handling="1",   # inconsistent: should be NA when MM=0
        )
    )

    # Dependent field must be corrected to NA.
    assert result.Missing_Handling == "NA", (
        f"Expected Missing_Handling='NA', got {result.Missing_Handling!r}"
    )

    # The flag must have been appended.
    assert "FLAG-QC-MISSING-LOGIC" in result.Extraction_Flags, (
        f"Expected FLAG-QC-MISSING-LOGIC in Extraction_Flags, got {result.Extraction_Flags!r}"
    )
