"""
Schema for the structured extraction produced by agent_2b_extractor.

All 65 Excel column names are declared as str fields with default "NR"
(Not Reported).  Internal tracking fields use different defaults.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, model_validator

# ---------------------------------------------------------------------------
# Excel-output field list (order matches final spreadsheet columns)
# ---------------------------------------------------------------------------

EXCEL_FIELDS: tuple[str, ...] = (
    # Cat 1 — Bibliographic
    "Paper_ID", "Authors", "Year", "Journal", "Title",
    # Cat 2 — Research design
    "RQ_Summary", "Num_Hypotheses", "Primary_Relationship", "Relationship_Direction",
    # Cat 3 — Dependent Variable
    "DV_Name", "DV_Construct", "DV_Measurement", "DV_Source", "DV_Type", "DV_Num",
    # Cat 4 — Independent Variable
    "IV_Name", "IV_Construct", "IV_Measurement", "IV_Source", "IV_Type", "IV_Num",
    # Cat 5 — Mediation
    "Mediator_Present", "Mediator_Name", "Mediator_Construct",
    "Mediator_Measurement", "Mediation_Method",
    # Cat 6 — Moderation
    "Moderator_Present", "Moderator_Name", "Moderator_Construct",
    "Moderator_Measurement", "Moderation_Method",
    # Cat 7 — Controls
    "Control_Num", "Control_List", "Control_Justified",
    # Cat 8 — Sample / Data
    "Sample_Size", "Sample_Context", "Data_Type", "Data_Source_Primary",
    "Unit_of_Analysis", "Time_Period",
    # Cat 9 — Model / Estimation
    "Model_Type", "Model_Type_Other", "Endogeneity_Addressed",
    "Endogeneity_Method", "Robustness_Checks",
    # Cat 10 — Missing Data
    "Missing_Mentioned", "Missing_Rate_Reported", "Missing_Rate_Value",
    "Missing_Variables", "Missing_Handling", "Missing_Handling_Other",
    "Missing_Justified", "Missing_Pattern_Tested", "Missing_Pattern_Result",
    "Missing_Sensitivity",
    # Cat 11 — Transparency / Replication
    "Data_Available", "Code_Available", "Software_Reported",
    "Software_Name", "Replication_Feasibility",
)

# ---------------------------------------------------------------------------
# Fields that must be forced to "NA" when Missing_Mentioned == "0"
# ---------------------------------------------------------------------------

_MISSING_DEPENDENTS: tuple[str, ...] = (
    "Missing_Rate_Reported",
    "Missing_Rate_Value",
    "Missing_Variables",
    "Missing_Handling",
    "Missing_Handling_Other",
    "Missing_Justified",
    "Missing_Pattern_Tested",
    "Missing_Pattern_Result",
    "Missing_Sensitivity",
)

# ---------------------------------------------------------------------------
# Coded-field valid-value sets (for warning-only range checks)
# ---------------------------------------------------------------------------

_MODEL_TYPE_VALID: frozenset[str] = frozenset(
    {str(i) for i in range(1, 12)} | {"NR", "NA"}
)
_DATA_TYPE_VALID: frozenset[str] = frozenset(
    {str(i) for i in range(1, 5)} | {"NR", "NA"}
)
_MISSING_HANDLING_VALID: frozenset[str] = frozenset(
    {str(i) for i in range(1, 11)} | {"NR", "NA"}
)
_BINARY_FIELDS: tuple[str, ...] = (
    "Missing_Mentioned", "Missing_Rate_Reported", "Missing_Justified",
    "Missing_Pattern_Tested", "Missing_Sensitivity",
    "Mediator_Present", "Moderator_Present",
    "Endogeneity_Addressed", "Code_Available", "Software_Reported",
)


class ExtractionResult(BaseModel):
    """Structured extraction result for a single paper.

    All 65 Excel fields default to "NR" (Not Reported).
    Internal tracking fields use separate defaults.
    """

    model_config = {"extra": "ignore"}

    # --- Cat 1: Bibliographic ---
    Paper_ID: str = "NR"
    Authors: str = "NR"
    Year: str = "NR"
    Journal: str = "NR"
    Title: str = "NR"

    # --- Cat 2: Research Design ---
    RQ_Summary: str = "NR"
    Num_Hypotheses: str = "NR"
    Primary_Relationship: str = "NR"
    Relationship_Direction: str = "NR"

    # --- Cat 3: Dependent Variable ---
    DV_Name: str = "NR"
    DV_Construct: str = "NR"
    DV_Measurement: str = "NR"
    DV_Source: str = "NR"
    DV_Type: str = "NR"
    DV_Num: str = "NR"

    # --- Cat 4: Independent Variable ---
    IV_Name: str = "NR"
    IV_Construct: str = "NR"
    IV_Measurement: str = "NR"
    IV_Source: str = "NR"
    IV_Type: str = "NR"
    IV_Num: str = "NR"

    # --- Cat 5: Mediation ---
    Mediator_Present: str = "NR"
    Mediator_Name: str = "NR"
    Mediator_Construct: str = "NR"
    Mediator_Measurement: str = "NR"
    Mediation_Method: str = "NR"

    # --- Cat 6: Moderation ---
    Moderator_Present: str = "NR"
    Moderator_Name: str = "NR"
    Moderator_Construct: str = "NR"
    Moderator_Measurement: str = "NR"
    Moderation_Method: str = "NR"

    # --- Cat 7: Controls ---
    Control_Num: str = "NR"
    Control_List: str = "NR"
    Control_Justified: str = "NR"

    # --- Cat 8: Sample / Data ---
    Sample_Size: str = "NR"
    Sample_Context: str = "NR"
    Data_Type: str = "NR"
    Data_Source_Primary: str = "NR"
    Unit_of_Analysis: str = "NR"
    Time_Period: str = "NR"

    # --- Cat 9: Model / Estimation ---
    Model_Type: str = "NR"
    Model_Type_Other: str = "NR"
    Endogeneity_Addressed: str = "NR"
    Endogeneity_Method: str = "NR"
    Robustness_Checks: str = "NR"

    # --- Cat 10: Missing Data ---
    Missing_Mentioned: str = "NR"
    Missing_Rate_Reported: str = "NR"
    Missing_Rate_Value: str = "NR"
    Missing_Variables: str = "NR"
    Missing_Handling: str = "NR"
    Missing_Handling_Other: str = "NR"
    Missing_Justified: str = "NR"
    Missing_Pattern_Tested: str = "NR"
    Missing_Pattern_Result: str = "NR"
    Missing_Sensitivity: str = "NR"

    # --- Cat 11: Transparency / Replication ---
    Data_Available: str = "NR"
    Code_Available: str = "NR"
    Software_Reported: str = "NR"
    Software_Name: str = "NR"
    Replication_Feasibility: str = "NR"

    # --- Tracking: page locators ---
    Cat1_Page: Optional[int] = None
    Cat3_DV_Page: Optional[int] = None
    Cat4_IV_Page: Optional[int] = None
    Cat8_Sample_Page: Optional[int] = None
    Cat10_Missing_Page: Optional[int] = None
    hypotheses_page: Optional[int] = None
    dv_page: Optional[int] = None
    iv_page: Optional[int] = None
    sample_page: Optional[int] = None
    missing_data_page: Optional[int] = None
    results_table_page: Optional[int] = None
    controls_page: Optional[int] = None
    appendix_page: Optional[int] = None

    # --- Tracking: quality / pipeline ---
    Extraction_Confidence: int = 2
    Extraction_Flags: str = ""
    Extraction_Notes: str = ""
    Time_Spent_Seconds: float = 0.0

    # -----------------------------------------------------------------------
    # Validators
    # -----------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _enforce_missing_consistency(cls, data: dict) -> dict:
        """Force dependent missing-data fields to 'NA' when Missing_Mentioned == '0'.

        Also appends FLAG-QC-MISSING-LOGIC to Extraction_Flags when a dependent
        field has a non-null, non-NA value despite Missing_Mentioned being '0'.
        """
        if isinstance(data, dict) and str(data.get("Missing_Mentioned", "")) == "0":
            # Detect inconsistency before correction.
            inconsistent = any(
                data.get(dep) not in (None, "NA", "NR", "")
                for dep in _MISSING_DEPENDENTS
            )
            if inconsistent:
                flag = "FLAG-QC-MISSING-LOGIC"
                existing = str(data.get("Extraction_Flags", "") or "")
                if flag not in existing:
                    data["Extraction_Flags"] = (
                        f"{existing} {flag}".strip() if existing else flag
                    )
            # Always correct dependents.
            for dep in _MISSING_DEPENDENTS:
                data[dep] = "NA"
        return data

    @model_validator(mode="after")
    def _coerce_and_warn(self) -> "ExtractionResult":
        """Coerce empty/None Excel fields to 'NR'; warn on out-of-range coded values."""
        # Coerce "" or None → "NR" for all Excel fields.
        for field_name in EXCEL_FIELDS:
            val = getattr(self, field_name, None)
            if val is None or val == "":
                object.__setattr__(self, field_name, "NR")

        # Warning-only range checks.
        model_type_val = getattr(self, "Model_Type", "NR")
        if model_type_val not in _MODEL_TYPE_VALID:
            warnings.warn(
                f"Model_Type={model_type_val!r} is out of expected range",
                UserWarning,
                stacklevel=2,
            )

        data_type_val = getattr(self, "Data_Type", "NR")
        if data_type_val not in _DATA_TYPE_VALID:
            warnings.warn(
                f"Data_Type={data_type_val!r} is out of expected range",
                UserWarning,
                stacklevel=2,
            )

        missing_handling_val = getattr(self, "Missing_Handling", "NR")
        if missing_handling_val not in _MISSING_HANDLING_VALID:
            warnings.warn(
                f"Missing_Handling={missing_handling_val!r} is out of expected range",
                UserWarning,
                stacklevel=2,
            )

        binary_valid = {"0", "1", "NR", "NA"}
        for bf in _BINARY_FIELDS:
            bval = getattr(self, bf, "NR")
            if bval not in binary_valid:
                warnings.warn(
                    f"{bf}={bval!r} is not a valid binary value (expected 0/1/NR/NA)",
                    UserWarning,
                    stacklevel=2,
                )

        return self

    # -----------------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain dict of all fields."""
        return self.model_dump()

    def to_excel_dict(self) -> dict:
        """Return only the 65 Excel output fields in column order."""
        d = self.model_dump()
        return {k: d[k] for k in EXCEL_FIELDS}

    def save(self, directory: Path) -> Path:
        """Serialise to JSON at *directory*/{Paper_ID}_extraction.json.

        Args:
            directory: Destination folder (created if absent).

        Returns:
            The path of the written file.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        out_path = directory / f"{self.Paper_ID}_extraction.json"
        out_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return out_path

    @classmethod
    def load(cls, path: Path) -> "ExtractionResult":
        """Load an ExtractionResult from a JSON file.

        Args:
            path: Path to the JSON file produced by :meth:`save`.

        Returns:
            A reconstituted :class:`ExtractionResult` instance.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
