"""
Agent 2b — Variable Extractor.

Sends targeted prompts to an LLM to extract the structured variables defined
in schemas/extraction_schema.py from the paper text.

Architecture: Step A (section locator) + B1-B4 (four extraction calls).

Usage (standalone runner)
--------------------------
    python agents/agent_2b_extractor.py [--force] [--parsed-dir PATH] [--extractions-dir PATH]
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import anthropic

# ---------------------------------------------------------------------------
# Path bootstrap — makes the module importable from any working directory.
# ---------------------------------------------------------------------------
_AGENTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _AGENTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402 — loads .env, creates data dirs
from schemas.parsed_paper import ParsedPaper  # noqa: E402
from schemas.extraction_schema import ExtractionResult, EXCEL_FIELDS  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_STEP_A_SYSTEM_PROMPT = """You are a precise document analyst.

Given the first portion of an academic paper's full text, identify the approximate page
number where each of the following sections begins.  Return a JSON object with exactly
these eight keys (integer page numbers or null if not found):

{
  "hypotheses_page": <int|null>,
  "dv_page": <int|null>,
  "iv_page": <int|null>,
  "sample_page": <int|null>,
  "missing_data_page": <int|null>,
  "results_table_page": <int|null>,
  "controls_page": <int|null>,
  "appendix_page": <int|null>
}

Return JSON only.  No markdown fences, no explanation.""".strip()

_STEP_B1_SYSTEM_PROMPT = """You are an expert research methodologist extracting structured variables
from management journal papers.

## TASK
Extract Categories 1-4 from the paper excerpts provided.

## OUTPUT FORMAT
Return a single JSON object with exactly the keys listed below.  No markdown fences.
All values must be strings.  Use "NR" (Not Reported) if information should exist but is
absent.  Use "NA" (Not Applicable) if the field is structurally absent for this paper.

## FIELDS TO EXTRACT

### Category 1 — Bibliographic
- Paper_ID: Already known — copy from the header line.
- Authors: Last names only, comma-separated.  If >4 authors, use "LastName1 et al."
- Year: Publication year (4 digits).  Extract from first-page header/footer.  "NR" if absent.
- Journal: Journal name.  Extract from header/footer.  "NR" if absent.
- Title: Full paper title.

### Category 2 — Research Design
- RQ_Summary: One sentence summarising the central research question.
- Num_Hypotheses: Count of formal hypotheses (H1, H2, H2a, H2b count as 3).
  Count sub-hypotheses separately.  Exclude Research Questions (RQ).  Give a number string.
- Primary_Relationship: "X → Y" notation of the main hypothesis (e.g. "CEO tenure → firm performance").
- Relationship_Direction: Hypothesized direction of the primary relationship (+/-/inverted-U/NR).
  Report the PREDICTED direction from the theory section, NOT the observed result.

### Category 3 — Dependent Variable
- DV_Name: Short name used in the paper (e.g. "ROA", "firm performance").
- DV_Construct: Theoretical construct the DV operationalises.
- DV_Measurement: SPECIFIC measurement formula + data source required.
  BAD: "measured using financial data"  GOOD: "net income / total assets from Compustat (2000-2020)"
- DV_Source: Primary data source for the DV (e.g., "Compustat", "hand-collected").
- DV_Type: "1" = continuous, "2" = dichotomous/binary, "3" = ordinal/count, "4" = other.  "NR" if unclear.
- DV_Num: Number of distinct DVs tested (string integer, e.g. "1", "2").

### Category 4 — Independent Variable (primary/focal IV only)
- IV_Name: Short name used in the paper.
- IV_Construct: Theoretical construct the IV operationalises.
- IV_Measurement: SPECIFIC measurement formula + data source required (same rule as DV).
- IV_Source: Primary data source for the IV.
- IV_Type: "1" = continuous, "2" = dichotomous, "3" = ordinal/count, "4" = other.  "NR" if unclear.
- IV_Num: Number of distinct focal IVs (string integer).

## CODING RULES
- NA = structurally absent (e.g. no mediator in a paper that doesn't test mediation).
- NR = should exist but is not reported in the text.
- Never return null or empty string; always use "NR" or "NA".

Return JSON only.""".strip()

_STEP_B2_SYSTEM_PROMPT = """You are an expert research methodologist extracting structured variables
from management journal papers.

## TASK
Extract Categories 5-7 from the paper excerpts provided.

## OUTPUT FORMAT
Return a single JSON object.  All values must be strings.  No markdown fences.
Use "NR" (Not Reported) or "NA" (Not Applicable) as appropriate.

## FIELDS TO EXTRACT

### Category 5 — Mediation
- Mediator_Present: "1" = yes, "0" = no.  "NR" if unclear.
- Mediator_Name: Short name(s) of mediator variable(s).  "NA" if Mediator_Present=0.
- Mediator_Construct: Theoretical construct(s).  "NA" if Mediator_Present=0.
- Mediator_Measurement: Specific measurement.  "NA" if Mediator_Present=0.
- Mediation_Method: Code for analytical method.
  "1" = Baron & Kenny; "2" = Sobel test; "3" = bootstrapping (Hayes PROCESS or similar);
  "4" = SEM path; "5" = other formal mediation test;
  "NA" if Mediator_Present=0; "NR" if unclear.

### Category 6 — Moderation
- Moderator_Present: "1" = yes, "0" = no.  "NR" if unclear.
- Moderator_Name: Short name(s) of moderator variable(s).  "NA" if Moderator_Present=0.
- Moderator_Construct: Theoretical construct(s).  "NA" if Moderator_Present=0.
- Moderator_Measurement: Specific measurement.  "NA" if Moderator_Present=0.
- Moderation_Method: Code for analytical method.
  "1" = interaction term in regression; "2" = subgroup analysis; "3" = other;
  "NA" if Moderator_Present=0; "NR" if unclear.

### Category 7 — Controls
- Control_Num: Count of control variables.  EXCLUDE all fixed effects (year FE, industry FE,
  firm FE, country FE do NOT count as control variables).  String integer or "NR".
- Control_List: Comma-separated list of control variable names.  "NR" if absent.
- Control_Justified: Was the inclusion of controls justified?
  "0" = No justification given;
  "1" = Partial justification (e.g., "following prior research [cite]" counts as PARTIAL, NOT full);
  "2" = Full justification (substantive rationale for each control);
  "NR" if no controls.

## CODING RULES
- NA = structurally absent for this paper.
- NR = should exist but is not reported.
- Never return null or empty string.

Return JSON only.""".strip()

_STEP_B3_SYSTEM_PROMPT = """You are an expert research methodologist extracting structured variables
from management journal papers.

## TASK
Extract Categories 8-9 from the paper excerpts provided.

## OUTPUT FORMAT
Return a single JSON object.  All values must be strings.  No markdown fences.
Use "NR" (Not Reported) or "NA" (Not Applicable) as appropriate.

## FIELDS TO EXTRACT

### Category 8 — Sample / Data
- Sample_Size: Final analytical N used in the MAIN REGRESSION TABLE (not the initial sample
  before exclusions).  String integer (e.g. "1423").  "NR" if not reported.
- Sample_Context: Industry/country/population description (e.g. "US public firms 2000-2015").
- Data_Type: "1" = cross-sectional; "2" = panel (multiple obs per unit over time);
  "3" = time-series; "4" = other/mixed.  "NR" if unclear.
- Data_Source_Primary: Primary dataset/database name (e.g. "Compustat", "CRSP", "survey").
- Unit_of_Analysis: Level at which observations are measured (e.g. "firm-year", "individual").
- Time_Period: Date range of data (e.g. "2005-2018").  "NR" if absent.

### Category 9 — Model / Estimation
- Model_Type: Numeric code for primary estimation method.
  "1" = OLS without fixed effects;
  "2" = Logit / Probit;
  "3" = OLS with fixed effects (includes OLS+FE, DiD — note DiD in Model_Type_Other);
  "4" = IV / 2SLS;
  "5" = Heckman selection model;
  "6" = GMM (Arellano-Bond, system GMM);
  "7" = Survival / Hazard (Cox, Weibull, etc.);
  "8" = SEM / path analysis / GSEM;
  "9" = HLM / multilevel model;
  "10" = Ordered / multinomial logit or probit;
  "11" = Tobit or censored regression;
  "NR" = not reported; "NA" = not applicable.
- Model_Type_Other: Free-text note on model variant (e.g. "DiD", "negative binomial").
  "NA" if nothing to note.
- Endogeneity_Addressed: "1" = yes, "0" = no.
  CRITICAL: Set to "1" ONLY if the paper EXPLICITLY uses the word "endogeneity" or
  "reverse causality" — do NOT infer from method alone.
- Endogeneity_Method: Description of endogeneity correction (e.g. "IV with lagged IV").
  "NA" if Endogeneity_Addressed=0.
- Robustness_Checks: "1" = robustness checks reported, "0" = none, "NR" if unclear.

## CODING RULES
- DiD studies: Model_Type = "3"; note "DiD" in Model_Type_Other.
- NA = structurally absent; NR = should exist but not reported.
- Never return null or empty string.

Return JSON only.""".strip()

_STEP_B4_SYSTEM_PROMPT = """You are an expert research methodologist extracting structured variables
from management journal papers.  This is the MOST CRITICAL section.

## TASK
Extract Categories 10-11 from the paper excerpts provided.

## OUTPUT FORMAT
Return a single JSON object.  All values must be strings except "flags_to_append" (list of strings).
No markdown fences.  Use "NR" or "NA" as appropriate.

## FIELDS TO EXTRACT

### Category 10 — Missing Data Handling (CRITICAL)
- Missing_Mentioned: "1" = yes (missing data is explicitly mentioned anywhere in the paper),
  "0" = no.
  IMPORTANT: Before coding "0", check the abstract, methods section, ALL footnotes,
  appendix, and notes below tables for ANY mention of missing data, listwise deletion,
  complete-case analysis, imputation, or sample attrition.
- Missing_Rate_Reported: "1" = a numeric rate (%) is reported, "0" = mentioned but no rate,
  "NA" if Missing_Mentioned=0.
- Missing_Rate_Value: The reported rate as a string (e.g. "12.3%").
  "NA" if not reported or Missing_Mentioned=0.
- Missing_Variables: Which variables have missing data (comma-separated).
  "NA" if not specified or Missing_Mentioned=0.
- Missing_Handling: Primary method used to handle missing data.
  "1" = listwise / complete-case deletion;
  "2" = mean/median/mode imputation;
  "3" = multiple imputation (MICE, Amelia, etc.);
  "4" = maximum likelihood (FIML, direct ML);
  "5" = regression imputation (single);
  "6" = hot-deck / nearest-neighbour imputation;
  "7" = interpolation (time-series);
  "8" = EM algorithm;
  "9" = pairwise deletion;
  "10" = other (describe in Missing_Handling_Other);
  "NA" if Missing_Mentioned=0; "NR" if mentioned but method unspecified.
- Missing_Handling_Other: Free-text if Missing_Handling="10".  "NA" otherwise.
- Missing_Justified: "1" = method choice substantively justified (e.g. MCAR assumption tested
  or strong statistical reasoning given — a bare citation does NOT count),
  "0" = not justified or only stated without reasoning,
  "NA" if Missing_Mentioned=0.
- Missing_Pattern_Tested: "1" = a formal statistical test of missingness pattern was conducted
  (e.g. Little's MCAR test, MAR test) — MUST be an actual test, not just a verbal claim,
  "0" = not tested, "NA" if Missing_Mentioned=0.
- Missing_Pattern_Result: Result of the pattern test (e.g. "MCAR p=0.43").
  "NA" if Missing_Pattern_Tested is not "1".
- Missing_Sensitivity: "1" = sensitivity analysis comparing results with/without imputed
  observations was conducted, "0" = not conducted, "NA" if Missing_Mentioned=0.

### Category 11 — Transparency / Replication
- Data_Available: "1" = data publicly available with DOI/URL, "2" = available on request,
  "3" = proprietary but source named (e.g. Bloomberg, Compustat), "0" = not available/unclear.
- Code_Available: "1" = analysis code publicly available, "0" = not available.
- Software_Reported: "1" = statistical software named, "0" = not reported.
- Software_Name: Name(s) of software (e.g. "Stata 17", "R").  "NA" if Software_Reported=0.
- Replication_Feasibility: "1" = feasible (requires Data_Available in {1,3} AND Code_Available=1),
  "0" = not feasible.

## FLAGS (add to "flags_to_append" list in your JSON response)
Include this key ONLY if triggered:
- FLAG-MISSING-AMBIGUOUS: Add if Missing_Mentioned coding was genuinely uncertain
  (e.g. found only ambiguous phrases like "complete data" without explicit discussion).
- FLAG-UNUSUAL-IMPUTATION: Add if Missing_Handling is a rare or non-standard method.

## CODING RULES
- If Missing_Mentioned="0", all dependent fields MUST be "NA".
- NA = structurally absent; NR = should exist but not reported.
- Never return null or empty string.

Return JSON only.""".strip()

# ---------------------------------------------------------------------------
# Fallback field tuples for each B step
# ---------------------------------------------------------------------------

_B1_FIELDS: tuple[str, ...] = (
    "Paper_ID", "Authors", "Year", "Journal", "Title",
    "RQ_Summary", "Num_Hypotheses", "Primary_Relationship", "Relationship_Direction",
    "DV_Name", "DV_Construct", "DV_Measurement", "DV_Source", "DV_Type", "DV_Num",
    "IV_Name", "IV_Construct", "IV_Measurement", "IV_Source", "IV_Type", "IV_Num",
)

_B2_FIELDS: tuple[str, ...] = (
    "Mediator_Present", "Mediator_Name", "Mediator_Construct",
    "Mediator_Measurement", "Mediation_Method",
    "Moderator_Present", "Moderator_Name", "Moderator_Construct",
    "Moderator_Measurement", "Moderation_Method",
    "Control_Num", "Control_List", "Control_Justified",
)

_B3_FIELDS: tuple[str, ...] = (
    "Sample_Size", "Sample_Context", "Data_Type", "Data_Source_Primary",
    "Unit_of_Analysis", "Time_Period",
    "Model_Type", "Model_Type_Other", "Endogeneity_Addressed",
    "Endogeneity_Method", "Robustness_Checks",
)

_B4_FIELDS: tuple[str, ...] = (
    "Missing_Mentioned", "Missing_Rate_Reported", "Missing_Rate_Value",
    "Missing_Variables", "Missing_Handling", "Missing_Handling_Other",
    "Missing_Justified", "Missing_Pattern_Tested", "Missing_Pattern_Result",
    "Missing_Sensitivity",
    "Data_Available", "Code_Available", "Software_Reported",
    "Software_Name", "Replication_Feasibility",
)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


_RETRY_QUEUE_NAME = "_retry_queue.jsonl"

def _is_rate_limit_error(exc_str: str) -> bool:
    s = (exc_str or "").lower()
    # Keep it simple + robust across SDK versions
    return ("429" in s) and ("rate" in s or "too many request" in s or "status" in s or "error" in s)

def _enqueue_retry(output_dir: Path, paper_id: str, flag: str, detail: str = "") -> None:
    """
    Append a retry record (JSONL). This survives crashes and doesn't block the run.
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / _RETRY_QUEUE_NAME
        rec = {
            "paper_id": paper_id,
            "flag": flag,
            "detail": (detail or "")[:300],
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Could not write retry queue for %s (%s).", paper_id, exc)

def _load_retry_queue_ids(extractions_dir: Path, retry_flags: list[str]) -> set[str]:
    """
    Return paper_ids found in _retry_queue.jsonl whose 'flag' matches any retry_flag.
    """
    ids: set[str] = set()
    path = Path(extractions_dir) / _RETRY_QUEUE_NAME
    if not path.exists():
        return ids

    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = str(rec.get("paper_id", "")).strip()
            flag = str(rec.get("flag", "")).strip()
            if not pid or not flag:
                continue
            if any(rf in flag for rf in retry_flags):
                ids.add(pid)
    except Exception as exc:
        logger.warning("Could not read retry queue %s (%s).", path, exc)

    return ids

def _scan_cached_extractions_for_flags(extractions_dir: Path, retry_flags: list[str]) -> set[str]:
    """
    Return paper_ids whose cached *_extraction.json has Extraction_Flags containing any retry_flag substring.
    """
    ids: set[str] = set()
    extractions_dir = Path(extractions_dir)

    for p in sorted(extractions_dir.glob("*_extraction.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            flags = str(d.get("Extraction_Flags", ""))
            if any(rf in flags for rf in retry_flags):
                ids.add(p.name.split("_extraction.json")[0])
        except Exception:
            continue

    return ids


def _extract_json(text: str) -> dict:
    """Attempt to parse a JSON object from a model response string.

    Tries three strategies in order:
    1. Direct ``json.loads`` on the stripped text.
    2. Strip markdown code fences (```json ... ```) and try again.
    3. Use a regex to find the first ``{ ... }`` block.

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If all three strategies fail.
    """
    stripped = text.strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", stripped)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"\{[\s\S]+\}", stripped)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from response: {stripped[:300]!r}")


def _call_llm(
    client: "anthropic.Anthropic",
    system: str,
    user_msg: str,
    max_tokens: int,
    paper_id: str,
    call_name: str,
) -> "tuple[str | None, int, int, int, str]":
    """
    Returns: (raw_text|None, input_tokens, output_tokens, duration_ms, exc_str).
    exc_str is "" on success.
    """
    exc_str: str = ""
    t0 = time.monotonic()
    duration_ms: int = 0
    try:
        ts = datetime.now().isoformat(timespec="seconds")
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        in_tok = response.usage.input_tokens or 0
        out_tok = response.usage.output_tokens or 0
        logger.info(
            "[%s] paper_id=%s call=%s input=%d output=%d duration_ms=%d",
            ts, paper_id, call_name, in_tok, out_tok, duration_ms,
        )
        return response.content[0].text, in_tok, out_tok, duration_ms, ""
    except Exception as exc:
        exc_str = str(exc)
        duration_ms = int((time.monotonic() - t0) * 1000)
        logger.error(
            "API error paper_id=%s call=%s duration_ms=%d: %s",
            paper_id, call_name, duration_ms, exc_str,
        )
        return None, 0, 0, duration_ms, exc_str



def _run_b_call(
    client: "anthropic.Anthropic",
    system: str,
    user_msg: str,
    call_name: str,
    paper_id: str,
    fallback_fields: "tuple[str, ...]",
) -> "tuple[dict, str]":
    """
    flags_string:
      - "" on success
      - "FLAG-PARSE-ERROR-{call_name}" when model responded but JSON couldn't be parsed (2 attempts)
      - "FLAG-RATE-LIMIT-{call_name}" when API was rate-limited (429) and retry also rate-limited
      - "FLAG-API-ERROR-{call_name}" for other API failures
    """
    # Attempt 1
    raw_text, _, _, _, _, exc1 = _call_llm(client, system, user_msg, 2048, paper_id, call_name)

    # If API failed (no text), handle rate limit vs other
    if raw_text is None:
        if _is_rate_limit_error(exc1):
            logger.warning("Rate limit on %s for %s; sleeping 60s then retrying.", call_name, paper_id)
            time.sleep(60)
        else:
            logger.warning("API failure on %s for %s; retrying once.", call_name, paper_id)

        raw_text2, _, _, _, _, exc2 = _call_llm(
            client, system, user_msg, 2048, paper_id, f"{call_name}-retry"
        )

        if raw_text2 is None:
            if _is_rate_limit_error(exc2):
                return {f: "NR" for f in fallback_fields}, f"FLAG-RATE-LIMIT-{call_name}"
            return {f: "NR" for f in fallback_fields}, f"FLAG-API-ERROR-{call_name}"

        # We got text on retry; parse once (no extra third call)
        try:
            return _extract_json(raw_text2), ""
        except ValueError as exc:
            logger.warning("Parse error on %s retry for %s: %s", call_name, paper_id, exc)
            return {f: "NR" for f in fallback_fields}, f"FLAG-PARSE-ERROR-{call_name}"

    # We got text on attempt 1; parse JSON
    try:
        return _extract_json(raw_text), ""
    except ValueError as exc:
        logger.warning("Parse error on %s first attempt for %s: %s", call_name, paper_id, exc)

    # Retry once with JSON reminder
    retry_msg = "Your previous response was not valid JSON. Return JSON only:\n\n" + user_msg
    raw_text2, _, _, _, _, exc2 = _call_llm(
        client, system, retry_msg, 2048, paper_id, f"{call_name}-retry"
    )

    if raw_text2 is None:
        if _is_rate_limit_error(exc2):
            return {f: "NR" for f in fallback_fields}, f"FLAG-RATE-LIMIT-{call_name}"
        return {f: "NR" for f in fallback_fields}, f"FLAG-API-ERROR-{call_name}"

    try:
        return _extract_json(raw_text2), ""
    except ValueError as exc:
        logger.warning("Parse error on %s retry for %s: %s", call_name, paper_id, exc)

    return {f: "NR" for f in fallback_fields}, f"FLAG-PARSE-ERROR-{call_name}"



def _run_step_a(paper: ParsedPaper, client: "anthropic.Anthropic") -> dict:
    """Locate key sections by page number.

    Returns all-None dict on any failure (non-critical step).
    """
    _ALL_NONE: dict = {
        "hypotheses_page": None,
        "dv_page": None,
        "iv_page": None,
        "sample_page": None,
        "missing_data_page": None,
        "results_table_page": None,
        "controls_page": None,
        "appendix_page": None,
    }

    exc_str: str = ""
    try:
        user_msg = (paper.full_text or "")[:8000]
        if not user_msg.strip():
            return dict(_ALL_NONE)

        raw_text, _, _, _, _ = _call_llm(
            client, _STEP_A_SYSTEM_PROMPT, user_msg, 256, paper.paper_id, "step-a"
        )
        if raw_text is None:
            return dict(_ALL_NONE)

        exc_str = ""
        try:
            parsed = _extract_json(raw_text)
        except ValueError as exc:
            exc_str = str(exc)
            logger.warning(
                "Step A parse failure for %s: %s", paper.paper_id, exc_str
            )
            return dict(_ALL_NONE)

        # Coerce all values to int | None.
        result: dict = {}
        for key in _ALL_NONE:
            coerce_exc: str = ""
            val = parsed.get(key)
            if val is None:
                result[key] = None
            else:
                try:
                    result[key] = int(val)
                except (TypeError, ValueError) as exc:
                    coerce_exc = str(exc)
                    result[key] = None
        return result

    except Exception as exc:
        exc_str = str(exc)
        logger.warning("Step A failed for %s: %s", paper.paper_id, exc_str)
        return dict(_ALL_NONE)


def _build_b1_message(paper: ParsedPaper) -> str:
    """Build the user-turn message for B1 (Categories 1-4)."""
    return (
        f"Paper ID: {paper.paper_id}\n"
        f"Title: {paper.title or '(empty)'}\n"
        f"Authors: {paper.authors or '(empty)'}\n\n"
        f"--- ABSTRACT ---\n{paper.abstract or '(empty)'}\n\n"
        f"--- THEORY / HYPOTHESES ---\n{paper.theory_hypotheses or '(empty)'}\n\n"
        f"--- METHODS / DATA (first 3000 chars) ---\n"
        f"{(paper.methods_data or '')[:3000] or '(empty)'}\n\n"
        f"Extract the JSON for Categories 1-4."
    )


def _build_b2_message(paper: ParsedPaper) -> str:
    """Build the user-turn message for B2 (Categories 5-7)."""
    return (
        f"Paper ID: {paper.paper_id}\n\n"
        f"--- THEORY / HYPOTHESES ---\n{paper.theory_hypotheses or '(empty)'}\n\n"
        f"--- METHODS / DATA (first 3000 chars) ---\n"
        f"{(paper.methods_data or '')[:3000] or '(empty)'}\n\n"
        f"--- RESULTS (first 1000 chars) ---\n"
        f"{(paper.results or '')[:1000] or '(empty)'}\n\n"
        f"Extract the JSON for Categories 5-7."
    )


def _build_b3_message(paper: ParsedPaper) -> str:
    """Build the user-turn message for B3 (Categories 8-9)."""
    return (
        f"Paper ID: {paper.paper_id}\n\n"
        f"--- METHODS / DATA ---\n{paper.methods_data or '(empty)'}\n\n"
        f"--- RESULTS (first 1500 chars) ---\n"
        f"{(paper.results or '')[:1500] or '(empty)'}\n\n"
        f"Extract the JSON for Categories 8-9."
    )


def _build_b4_message(paper: ParsedPaper) -> str:
    """Build the user-turn message for B4 (Categories 10-11)."""
    return (
        f"Paper ID: {paper.paper_id}\n\n"
        f"--- METHODS / DATA ---\n{paper.methods_data or '(empty)'}"
        f"\n---\n"
        f"--- FOOTNOTES ---\n{paper.footnotes or '(empty)'}"
        f"\n---\n"
        f"--- APPENDIX (first 2000 chars) ---\n"
        f"{(paper.appendix or '')[:2000] or '(empty)'}"
        f"\n---\n"
        f"--- RESULTS (last 500 chars) ---\n"
        f"{(paper.results or '')[-500:] or '(empty)'}\n\n"
        f"Extract the JSON for Categories 10-11."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_paper(
    parsed_paper: ParsedPaper,
    client: "anthropic.Anthropic",
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> dict:
    """Extract structured variables from a single parsed paper.

    Runs Step A (section locator) and Steps B1-B4 (extraction calls).
    Caches the result: if an extraction JSON already exists and *force* is False,
    the cached result is returned without making any API calls.

    Args:
        parsed_paper: A :class:`~schemas.parsed_paper.ParsedPaper` instance.
        client: An initialised Anthropic SDK client.
        output_dir: Directory where extraction JSONs are saved.
            Defaults to ``config.EXTRACTIONS_DIR``.
        force: If True, re-extract even if a cached result exists.

    Returns:
        A dict with all EXCEL_FIELDS plus internal tracking fields.
    """
    if output_dir is None:
        output_dir = config.EXTRACTIONS_DIR

    paper_id = parsed_paper.paper_id
    output_dir = Path(output_dir)
    cache_path = output_dir / f"{paper_id}_extraction.json"

    # --- Cache check ---
    if not force and cache_path.exists():
        logger.info("Cache hit for %s — loading existing extraction.", paper_id)
        cache_exc: str = ""
        try:
            return ExtractionResult.load(cache_path).to_dict()
        except Exception as exc:
            cache_exc = str(exc)
            logger.warning(
                "Failed to load cached extraction for %s (%s); re-extracting.",
                paper_id, cache_exc,
            )

    t_start = time.monotonic()
    all_flags: list[str] = []
    parse_errors: int = 0

    # --- Step A: locate sections ---
    locator = _run_step_a(parsed_paper, client)

    # --- Step B1: Categories 1-4 ---
    b1_msg = _build_b1_message(parsed_paper)
    b1_result, b1_flag = _run_b_call(
        client, _STEP_B1_SYSTEM_PROMPT, b1_msg, "b1", paper_id, _B1_FIELDS
    )
    if b1_flag:
        all_flags.append(b1_flag)
        parse_errors += 1

    # --- Step B2: Categories 5-7 ---
    b2_msg = _build_b2_message(parsed_paper)
    b2_result, b2_flag = _run_b_call(
        client, _STEP_B2_SYSTEM_PROMPT, b2_msg, "b2", paper_id, _B2_FIELDS
    )
    if b2_flag:
        all_flags.append(b2_flag)
        parse_errors += 1

    # --- Step B3: Categories 8-9 ---
    b3_msg = _build_b3_message(parsed_paper)
    b3_result, b3_flag = _run_b_call(
        client, _STEP_B3_SYSTEM_PROMPT, b3_msg, "b3", paper_id, _B3_FIELDS
    )
    if b3_flag:
        all_flags.append(b3_flag)
        parse_errors += 1

    # --- Step B4: Categories 10-11 ---
    b4_msg = _build_b4_message(parsed_paper)
    b4_result, b4_flag = _run_b_call(
        client, _STEP_B4_SYSTEM_PROMPT, b4_msg, "b4", paper_id, _B4_FIELDS
    )
    if b4_flag:
        all_flags.append(b4_flag)
        parse_errors += 1

    # Pop flags_to_append BEFORE merge (not a valid Pydantic field).
    b4_extra_flags: list = b4_result.pop("flags_to_append", [])
    if isinstance(b4_extra_flags, list):
        all_flags.extend(b4_extra_flags)

    # --- Merge all dicts ---
    merged: dict = {}
    merged.update(b1_result)
    merged.update(b2_result)
    merged.update(b3_result)
    merged.update(b4_result)

    # Always overwrite Paper_ID with the authoritative value.
    merged["Paper_ID"] = paper_id

    # --- Missing consistency (belt-and-suspenders before Pydantic validator) ---
    if merged.get("Missing_Mentioned") == "0":
        for dep in (
            "Missing_Rate_Reported", "Missing_Rate_Value", "Missing_Variables",
            "Missing_Handling", "Missing_Handling_Other", "Missing_Justified",
            "Missing_Pattern_Tested", "Missing_Pattern_Result", "Missing_Sensitivity",
        ):
            merged[dep] = "NA"

    # --- Add locator fields ---
    merged.update(locator)
    # Map locator fields to Cat*_Page convenience aliases.
    merged["Cat1_Page"] = None  # bibliographic fields have no single source page
    merged["Cat3_DV_Page"] = locator.get("dv_page")
    merged["Cat4_IV_Page"] = locator.get("iv_page")
    merged["Cat8_Sample_Page"] = locator.get("sample_page")
    merged["Cat10_Missing_Page"] = locator.get("missing_data_page")

    # --- Add tracking fields ---
    elapsed = time.monotonic() - t_start
    merged["Extraction_Confidence"] = 1 if parse_errors > 0 else 2
    merged["Extraction_Flags"] = " ".join(all_flags)
    merged["Extraction_Notes"] = ""
    merged["Time_Spent_Seconds"] = round(elapsed, 2)

    # --- Pydantic validation ---
    exc_str: str = ""
    try:
        result_obj = ExtractionResult(**merged)
    except Exception as exc:
        exc_str = str(exc)
        logger.error(
            "Pydantic validation failed for %s: %s. Constructing minimal result.",
            paper_id, exc_str,
        )
        result_obj = ExtractionResult(
            Paper_ID=paper_id,
            Extraction_Flags=f"FLAG-PYDANTIC-ERROR {exc_str[:200]}",
            Extraction_Confidence=1,
            Time_Spent_Seconds=round(elapsed, 2),
        )

    # --- Save ---
    # --- Layer 2: Do NOT save if we were rate-limited (prevents caching NR outputs) ---
    flags_str = str(result_obj.Extraction_Flags or "")
    if "FLAG-RATE-LIMIT-" in flags_str:
        logger.warning(
            "Not saving extraction for %s due to rate-limit flags: %s",
            paper_id, flags_str
        )
        # Mark for retry
        for f in all_flags:
            if f.startswith("FLAG-RATE-LIMIT-"):
                _enqueue_retry(output_dir, paper_id, f, detail="Skipped save due to rate-limit.")
        return result_obj.to_dict()

    # --- Save (normal path) ---
    save_exc: str = ""
    try:
        saved_path = result_obj.save(output_dir)
        logger.info("Saved extraction for %s → %s", paper_id, saved_path)
    except Exception as exc:
        save_exc = str(exc)
        logger.error("Failed to save extraction for %s: %s", paper_id, save_exc)

    return result_obj.to_dict()



def extract_all_papers(
    parsed_dir: Optional[Path] = None,
    extractions_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    force: bool = False,
    eligible_codes: tuple = ("EQR", "MM"),
    retry_flags: Optional[list[str]] = None,
) -> list[dict]:
    """Extract variables for all eligible papers in *parsed_dir*.

    For each paper, checks that a classification JSON exists in *extractions_dir*
    and that the Classification_Code is in *eligible_codes*.  Skips ineligible papers.

    Args:
        parsed_dir: Folder containing ``{paper_id}.json`` ParsedPaper files.
            Defaults to ``config.PARSED_DIR``.
        extractions_dir: Destination for extraction JSONs.
            Defaults to ``config.EXTRACTIONS_DIR``.
        api_key: Anthropic API key.  Defaults to ``config.ANTHROPIC_API_KEY``.
        force: Re-extract even if cache exists.
        eligible_codes: Classification codes eligible for full extraction.

    Returns:
        List of extraction result dicts (one per paper processed).
    """
    import anthropic as anthropic_lib  # lazy import

    from schemas.classification_schema import ClassificationResult  # noqa: E402

    if parsed_dir is None:
        parsed_dir = config.PARSED_DIR
    if extractions_dir is None:
        extractions_dir = config.EXTRACTIONS_DIR
    if api_key is None:
        api_key = config.ANTHROPIC_API_KEY

    parsed_dir = Path(parsed_dir)
    extractions_dir = Path(extractions_dir)

    if not api_key:
        logger.error(
            "ANTHROPIC_API_KEY is not set. Cannot extract papers. "
            "Add it to .env or set the environment variable."
        )
        return []

    # Find all parsed paper JSONs — exclude *_classification.json and *_extraction.json.
    json_files = sorted(
        f for f in parsed_dir.glob("*.json")
        if not f.name.endswith("_classification.json")
        and not f.name.endswith("_extraction.json")
    )

    if not json_files:
        logger.warning("No parsed paper JSONs found in %s", parsed_dir)
        return []

    client = anthropic_lib.Anthropic(api_key=api_key)
    
    # If retry_flags is set, only rerun affected paper_ids and force re-extraction.
    target_ids: Optional[set[str]] = None
    if retry_flags:
        retry_flags = [str(x).strip() for x in retry_flags if str(x).strip()]
        target_ids = set()
        target_ids |= _scan_cached_extractions_for_flags(extractions_dir, retry_flags)
        target_ids |= _load_retry_queue_ids(extractions_dir, retry_flags)

        if not target_ids:
            logger.warning("retry_flags provided but no matching papers found: %s", retry_flags)
            return []
        logger.info("Retry mode: rerunning %d paper(s) for flags=%s", len(target_ids), retry_flags)
    
    results: list[dict] = []

    for jf in json_files:
        paperid = jf.stem.split(" - ")[0].strip()

        # Check classification exists.
        cls_path = extractions_dir / f"{paperid}_classification.json"
        if not cls_path.exists():
            logger.info("No classification for %s — skipping.", paperid)
            continue

        # Load classification.
        cls_exc: str = ""
        try:
            cls_result = ClassificationResult.load(cls_path)
        except Exception as exc:
            cls_exc = str(exc)
            logger.error("Cannot load classification for %s: %s", paperid, cls_exc)
            continue

        # Check eligibility.
        if cls_result.Classification_Code not in eligible_codes:
            logger.info(
                "Skipping %s — Classification_Code=%s not in eligible_codes=%s.",
                paperid, cls_result.Classification_Code, eligible_codes,
            )
            continue

        # Load ParsedPaper.
        load_exc: str = ""
        try:
            paper = ParsedPaper.load(jf)
        except Exception as exc:
            load_exc = str(exc)
            logger.error("Cannot load ParsedPaper from %s: %s", jf, load_exc)
            continue

        if target_ids is not None and paperid not in target_ids:
            continue

        # In retry mode, always force re-extraction even if cache exists.
        effective_force = True if target_ids is not None else force
        result = extract_paper(paper, client, output_dir=extractions_dir, force=effective_force)        
        
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def _print_summary_table(results: list[dict]) -> None:
    """Print a formatted summary table of extraction results."""
    if not results:
        print("No results to display.")
        return

    col_paper = max(8, max(len(str(r.get("Paper_ID", ""))) for r in results))
    col_mh = 16
    col_conf = 5
    col_flags = 60
    sep_len = col_paper + col_mh + col_conf + col_flags + 10

    header = (
        f"{'Paper_ID':<{col_paper}}  "
        f"{'Missing_Handling':<{col_mh}}  "
        f"{'Conf':<{col_conf}}  "
        f"Flags"
    )
    sep = "-" * sep_len

    print("\n" + "=" * sep_len)
    print("EXTRACTION SUMMARY")
    print("=" * sep_len)
    print(header)
    print(sep)

    for r in sorted(results, key=lambda x: x.get("Paper_ID", "")):
        pid = str(r.get("Paper_ID", ""))
        mh = str(r.get("Missing_Handling", ""))
        conf = str(r.get("Extraction_Confidence", ""))
        flags = str(r.get("Extraction_Flags", ""))[:col_flags]
        print(f"{pid:<{col_paper}}  {mh:<{col_mh}}  {conf:<{col_conf}}  {flags}")

    print(sep)
    print(f"\nTotal: {len(results)} paper(s)")
    print("=" * sep_len + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract structured variables from eligible papers in data/parsed/."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if a cached result exists.",
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=None,
        help="Override the parsed papers directory.",
    )
    parser.add_argument(
        "--extractions-dir",
        type=Path,
        default=None,
        help="Override the extractions output directory.",
    )  
    parser.add_argument(
    "--retry-flags",
    nargs="*",
    default=None,
    help="Rerun only papers affected by these flags (e.g. FLAG-PARSE-ERROR-b4 FLAG-RATE-LIMIT-b4).",
    )
    args = parser.parse_args()

    results = extract_all_papers(
        parsed_dir=args.parsed_dir,
        extractions_dir=args.extractions_dir,
        force=args.force,
        retry_flags=args.retry_flags,
    )

    if results:
        _print_summary_table(results)

        # Assert: no EXCEL_FIELDS value should be None in any result.
        failures: list[str] = []
        for r in results:
            for field_name in EXCEL_FIELDS:
                if r.get(field_name) is None:
                    failures.append(f"{r.get('Paper_ID', '?')}.{field_name} is None")

        if failures:
            print("ASSERTION FAILURES:")
            for f in failures:
                print(f"  {f}")
            sys.exit(1)
        else:
            print("All required fields are non-None. Assertions passed.")

        # Verify both classification and extraction files exist for each processed paper.
        output_dir = (
            Path(args.extractions_dir) if args.extractions_dir else config.EXTRACTIONS_DIR
        )
        missing_pairs: list[str] = []
        for r in results:
            pid = r.get("Paper_ID", "")
            cls_f = output_dir / f"{pid}_classification.json"
            ext_f = output_dir / f"{pid}_extraction.json"
            if not cls_f.exists():
                missing_pairs.append(f"{pid}: missing _classification.json")
            if not ext_f.exists():
                missing_pairs.append(f"{pid}: missing _extraction.json")

        if missing_pairs:
            print("MISSING FILES:")
            for mf in missing_pairs:
                print(f"  {mf}")
        else:
            print(
                "All processed papers have both _classification.json and _extraction.json."
            )
    else:
        print(
            "\nNo papers were extracted.\n"
            "  • Ensure data/parsed/ contains at least one {paper_id}.json file\n"
            "    with a corresponding _classification.json in data/extractions/.\n"
            "  • Ensure the paper's Classification_Code is EQR or MM.\n"
            "  • Ensure ANTHROPIC_API_KEY is set in your .env file.\n"
        )
