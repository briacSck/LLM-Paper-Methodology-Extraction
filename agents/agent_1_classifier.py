"""
Agent 1 — Paper Classifier.

Reads each ParsedPaper JSON from data/parsed/ and assigns it one of seven
mutually-exclusive category codes using a two-run Claude call for stability
checking.

No PDF text is re-read; only title, abstract, and first 2000 chars of
methods_data are sent to the LLM (token budget ≤ 4000 for paper content).

Usage (standalone runner)
--------------------------
    python agents/agent_1_classifier.py

This classifies all papers in data/parsed/ that do not yet have a
classification JSON in data/extractions/, then prints a summary table.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Path bootstrap — makes the module importable from any working directory.
# ---------------------------------------------------------------------------
_AGENTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _AGENTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import anthropic  # noqa: E402

import config  # noqa: E402  — loads .env, creates data dirs
from schemas.parsed_paper import ParsedPaper  # noqa: E402
from schemas.classification_schema import (  # noqa: E402
    ClassificationResult,
    ELIGIBLE_MAP,
    VALID_CODES,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Classification framework — embedded verbatim in the system prompt.
# ---------------------------------------------------------------------------

_CLASSIFICATION_FRAMEWORK = """
## CLASSIFICATION FRAMEWORK

### CATEGORIES (mutually exclusive — apply in the order listed below)

**EQR** — Empirical Quantitative – Regression
Hypotheses are tested via a regression-type statistical model.
ML/NLP may be used UPSTREAM to construct variables; the hypothesis test itself
must be regression-based.
Includes: OLS, Logit/Probit, Fixed Effects, Difference-in-Differences (DiD),
Structural Equation Modeling (SEM), GSEM, Survival/Hazard models (Cox, Weibull),
Hierarchical Linear Modeling (HLM), GMM, Heckman selection models.
→ Eligible for full variable extraction.

**EQNR-ML** — ML / Predictive
ML model performance (accuracy, F1, AUC, loss) IS the primary reported finding.
No downstream regression hypothesis testing is performed.
→ NOT eligible.

**EQNR-Other** — Quantitative, non-regression, non-ML
Includes: pure descriptive statistics, manual measurement studies, pure topic
modeling with no regression output, event studies with only raw CARs reported.
→ NOT eligible.

**MM** — Mixed Methods
Contains BOTH a qualitative component AND a self-contained quantitative
regression component.
→ Conditional. Eligible only if the regression component has its own hypotheses.

**EQL** — Empirical Qualitative
Interviews, ethnography, grounded theory, or case studies only.
→ NOT eligible.

**CT** — Conceptual / Theoretical / Methodological
No original hypothesis testing. Includes: literature reviews, theory papers,
"how-to ML" method showcases, measure construction papers without hypothesis
testing.
→ NOT eligible.

**MA** — Meta-Analysis / Systematic Review
Unit of analysis = prior studies, not individual observations.
Includes MASEM (even though it reports SEM paths).
→ NOT eligible.

---

### KEY DECISION RULES (apply in this exact order)

1. Does ML/NLP construct a variable that then enters a regression hypothesis
   test? → **EQR** (ML is upstream, not the primary result)
2. Is ML model accuracy / F1 / AUC the reported key result? → **EQNR-ML**
3. Are DiD, SEM, GSEM, survival models, or ordered probit the primary test?
   → **EQR**
4. Is pure LDA / topic modeling used with no downstream regression? → **EQNR-Other**
5. Does the paper only demonstrate a method without testing management
   hypotheses? → **CT**
6. Does the paper synthesise prior studies as its unit of analysis? → **MA**

---

### CONFIDENCE SCALE

4 = Classification is unambiguous from the abstract alone.
3 = Required reading the methods section to confirm.
2 = Applied a decision rule; the classification is contestable.
1 = Cannot classify even after reading the full excerpt → MANDATORY FLAG.

---

### FLAGS (add all that apply; separate with commas)

[FLAG-MULTISTUDY]              — 2+ distinct empirical studies in one paper.
[FLAG-ML-UPSTREAM]             — ML constructs IV/DV; verify regression is the
                                  hypothesis test.
[FLAG-BAYESIAN]                — Bayesian inference; verify management hypotheses
                                  are tested directionally.
[FLAG-CLASSIFICATION-UNSTABLE] — Two classification runs gave different codes
                                  (set by pipeline, not by you).
[FLAG-INCOMPLETE-SOURCE]       — Methods section is empty or truncated.
""".strip()

_SYSTEM_PROMPT = f"""You are an expert research methodologist classifying academic management
journal papers by their primary empirical strategy.

{_CLASSIFICATION_FRAMEWORK}

---

### OUTPUT INSTRUCTIONS

Return a **single JSON object** and nothing else — no markdown fences, no
explanatory text before or after the object.  The object must have exactly
these seven keys:

{{
  "Paper_ID": "<paper id string>",
  "Classification_Code": "<EQR|EQNR-ML|EQNR-Other|MM|EQL|CT|MA>",
  "Eligible_Extraction": "<Yes|No|Conditional>",
  "Confidence_Score": <1|2|3|4>,
  "Classification_Basis": "<1-2 sentences quoting or citing specific evidence from the paper>",
  "Flag": "<comma-separated flags, or empty string>",
  "Requires_Human_Review": <true|false>
}}

Set Requires_Human_Review to true if Confidence_Score is 1, if the paper is
flagged [FLAG-INCOMPLETE-SOURCE], or if you are genuinely uncertain.
""".strip()

# Max characters of methods_data to include in the prompt.
_METHODS_PREVIEW_CHARS = 2000

# Max characters for the abstract (guard against unusually long abstracts).
_ABSTRACT_MAX_CHARS = 3000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_user_message(paper: ParsedPaper) -> str:
    """Construct the user-turn content sent to the classifier.

    Includes title, authors, abstract (capped), and methods preview.
    Total paper content is intentionally kept under ~4000 tokens.

    Args:
        paper: The parsed paper to classify.

    Returns:
        A formatted string ready to send as a ``user`` message.
    """
    abstract = (paper.abstract or "").strip()
    if not abstract:
        # Fall back to the first 800 chars of full_text if abstract is missing.
        abstract = paper.full_text[:800].strip()

    abstract = abstract[:_ABSTRACT_MAX_CHARS]
    methods_preview = (paper.methods_data or "")[:_METHODS_PREVIEW_CHARS].strip()

    flags_hint = ""
    if not methods_preview:
        flags_hint = (
            "\nNOTE: The methods section could not be extracted from this PDF. "
            "Apply [FLAG-INCOMPLETE-SOURCE] and classify based on abstract alone."
        )

    return (
        f"Paper ID: {paper.paper_id}\n"
        f"Title: {paper.title or '(unknown)'}\n"
        f"Authors: {paper.authors or '(unknown)'}\n\n"
        f"--- ABSTRACT ---\n{abstract}\n\n"
        f"--- METHODS / DATA SECTION (first {_METHODS_PREVIEW_CHARS} chars) ---\n"
        f"{methods_preview or '(empty)'}"
        f"{flags_hint}\n\n"
        f"Classify this paper.  Return the JSON object only."
    )


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

    # Strategy 1: direct parse.
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences.
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", stripped)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: find first {...} block.
    brace_match = re.search(r"\{[\s\S]+\}", stripped)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from response: {stripped[:300]!r}")


def _validate_and_normalise(raw: dict, paper_id: str) -> dict:
    """Validate a parsed LLM response dict and normalise values.

    - Coerces Confidence_Score to int.
    - Forces Eligible_Extraction to the deterministic value from ELIGIBLE_MAP.
    - Ensures Classification_Code is a valid code (defaults to "CT" if not).
    - Ensures Flag is a string.
    - Ensures Requires_Human_Review is a bool.

    Args:
        raw: Dict from :func:`_extract_json`.
        paper_id: Used to populate Paper_ID if missing.

    Returns:
        A normalised dict ready for :class:`ClassificationResult`.
    """
    code = str(raw.get("Classification_Code", "CT")).strip()
    if code not in VALID_CODES:
        logger.warning(
            "Invalid Classification_Code %r for %s; defaulting to CT.", code, paper_id
        )
        code = "CT"

    confidence = int(raw.get("Confidence_Score", 1))
    confidence = max(1, min(4, confidence))

    eligible = ELIGIBLE_MAP[code]  # deterministic override

    flag_raw = raw.get("Flag", "")
    flag = str(flag_raw).strip() if flag_raw else ""

    review = bool(raw.get("Requires_Human_Review", False))
    # Force review if confidence is 1 or flag includes incomplete source.
    if confidence == 1 or "[FLAG-INCOMPLETE-SOURCE]" in flag:
        review = True

    return {
        "Paper_ID": str(raw.get("Paper_ID", paper_id)),
        "Classification_Code": code,
        "Eligible_Extraction": eligible,
        "Confidence_Score": confidence,
        "Classification_Basis": str(raw.get("Classification_Basis", "")).strip(),
        "Flag": flag,
        "Requires_Human_Review": review,
    }


def _call_claude(
    client: anthropic.Anthropic,
    user_message: str,
    paper_id: str,
    run_number: int,
    temperature: float,
) -> tuple[dict, int]:
    """Make a single classification call to Claude and parse the response.

    Retries once with a JSON-reminder prepend if the first parse fails.

    Args:
        client: Anthropic SDK client.
        user_message: The formatted user-turn content.
        paper_id: Used only for logging.
        run_number: 1 or 2, for logging.
        temperature: Sampling temperature.

    Returns:
        Tuple of (parsed_normalised_dict, total_tokens_used).
        On double failure returns a default error dict and 0 tokens.
    """
    def _attempt(msg: str) -> tuple[dict | None, int]:
        """Inner attempt: call API, parse response, return (dict|None, tokens)."""
        try:
            ts = datetime.now().isoformat(timespec="seconds")
            response = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=512,
                temperature=temperature,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": msg}],
            )
            tokens = (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)
            logger.info(
                "[%s] paper_id=%s run=%d temp=%.1f tokens=%d",
                ts,
                paper_id,
                run_number,
                temperature,
                tokens,
            )
            raw_text = response.content[0].text
            parsed = _extract_json(raw_text)
            normalised = _validate_and_normalise(parsed, paper_id)
            return normalised, tokens
        except (anthropic.APIError, anthropic.APIConnectionError) as api_err:
            logger.error(
                "API error for %s run %d: %s", paper_id, run_number, api_err
            )
            return None, 0
        except ValueError as parse_err:
            logger.warning(
                "JSON parse error for %s run %d: %s", paper_id, run_number, parse_err
            )
            return None, 0
        except Exception as exc:
            logger.error(
                "Unexpected error for %s run %d: %s", paper_id, run_number, exc
            )
            return None, 0

    # First attempt.
    result, tokens = _attempt(user_message)
    if result is not None:
        return result, tokens

    # Single retry with JSON reminder prepended.
    logger.info("Retrying %s run %d with JSON reminder.", paper_id, run_number)
    retry_msg = (
        "Your previous response was not valid JSON. Return JSON only:\n\n"
        + user_message
    )
    result, tokens = _attempt(retry_msg)
    if result is not None:
        return result, tokens

    # Both attempts failed — return a safe default.
    logger.error(
        "Both attempts failed for %s run %d. Returning default CT result.",
        paper_id,
        run_number,
    )
    default: dict = {
        "Paper_ID": paper_id,
        "Classification_Code": "CT",
        "Eligible_Extraction": "No",
        "Confidence_Score": 1,
        "Classification_Basis": "Classification failed due to API/parse error.",
        "Flag": "[FLAG-PARSE-ERROR]",
        "Requires_Human_Review": True,
    }
    return default, 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_paper(
    parsed_paper: ParsedPaper,
    client: anthropic.Anthropic,
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> dict:
    """Classify a single parsed paper and return the result dict.

    Runs Claude twice (temperature=0 and temperature=0.3).  If the two runs
    disagree on the Classification_Code the result is flagged as unstable and
    Requires_Human_Review is set to True.

    Caches the result: if a classification JSON already exists in *output_dir*
    and *force* is False, the cached result is returned without making any API
    calls.

    Args:
        parsed_paper: A :class:`~schemas.parsed_paper.ParsedPaper` instance.
        client: An initialised :class:`anthropic.Anthropic` client.
        output_dir: Directory where classification JSONs are saved.
            Defaults to ``config.EXTRACTIONS_DIR``.
        force: If True, re-classify even if a cached result exists.

    Returns:
        A dict with keys matching the Excel column names plus internal keys
        ``run1_code``, ``run2_code``, ``parse_quality_note``.
    """
    if output_dir is None:
        output_dir = config.EXTRACTIONS_DIR

    paper_id = parsed_paper.paper_id
    cache_path = Path(output_dir) / f"{paper_id}_classification.json"

    # --- Cache check ---
    if not force and cache_path.exists():
        logger.info("Cache hit for %s — loading existing classification.", paper_id)
        try:
            return ClassificationResult.load(cache_path).to_dict()
        except Exception as exc:
            logger.warning(
                "Failed to load cached classification for %s (%s); re-classifying.",
                paper_id,
                exc,
            )

    # --- Build prompt content ---
    user_message = _build_user_message(parsed_paper)

    # --- Run 1: temperature=0 (deterministic) ---
    result1, _tokens1 = _call_claude(
        client, user_message, paper_id, run_number=1, temperature=0.0
    )

    # --- Run 2: temperature=0.3 (slight variation) ---
    result2, _tokens2 = _call_claude(
        client, user_message, paper_id, run_number=2, temperature=0.3
    )

    # --- Stability check ---
    code1 = result1["Classification_Code"]
    code2 = result2["Classification_Code"]
    codes_differ = code1 != code2

    if codes_differ:
        logger.warning(
            "Classification unstable for %s: run1=%s run2=%s. "
            "Using run1; flagging for human review.",
            paper_id,
            code1,
            code2,
        )

    # Use run 1 as the authoritative result.
    final = dict(result1)

    # Append instability flag if needed.
    if codes_differ:
        existing_flags = final["Flag"]
        unstable_flag = "[FLAG-CLASSIFICATION-UNSTABLE]"
        if unstable_flag not in existing_flags:
            final["Flag"] = (
                f"{existing_flags}, {unstable_flag}".lstrip(", ")
                if existing_flags
                else unstable_flag
            )
        final["Requires_Human_Review"] = True

    # Add FLAG-INCOMPLETE-SOURCE if methods section was empty.
    if not (parsed_paper.methods_data or "").strip():
        incomplete_flag = "[FLAG-INCOMPLETE-SOURCE]"
        if incomplete_flag not in final["Flag"]:
            final["Flag"] = (
                f"{final['Flag']}, {incomplete_flag}".lstrip(", ")
                if final["Flag"]
                else incomplete_flag
            )
        final["Requires_Human_Review"] = True

    # --- Assemble ClassificationResult ---
    cr = ClassificationResult(
        Paper_ID=final["Paper_ID"],
        Classification_Code=final["Classification_Code"],
        Eligible_Extraction=final["Eligible_Extraction"],
        Confidence_Score=final["Confidence_Score"],
        Classification_Basis=final["Classification_Basis"],
        Flag=final["Flag"],
        Requires_Human_Review=final["Requires_Human_Review"],
        run1_code=code1,
        run2_code=code2,
        parse_quality_note=parsed_paper.parse_notes,
    )

    # --- Persist ---
    try:
        saved_path = cr.save(output_dir)
        logger.info("Saved classification for %s → %s", paper_id, saved_path)
    except Exception as exc:
        logger.error("Failed to save classification for %s: %s", paper_id, exc)

    return cr.to_dict()


def classify_all_papers(
    parsed_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    force: bool = False,
) -> list[dict]:
    """Classify every ParsedPaper JSON in *parsed_dir*.

    Skips papers that already have a classification JSON in *output_dir*
    (unless *force* is True).

    Args:
        parsed_dir: Folder containing ``{paper_id}.json`` ParsedPaper files.
            Defaults to ``config.PARSED_DIR``.
        output_dir: Destination for classification JSONs.
            Defaults to ``config.EXTRACTIONS_DIR``.
        api_key: Anthropic API key.  Defaults to ``config.ANTHROPIC_API_KEY``.
        force: Re-classify even if cache exists.

    Returns:
        List of classification result dicts (one per paper processed).
    """
    if parsed_dir is None:
        parsed_dir = config.PARSED_DIR
    if output_dir is None:
        output_dir = config.EXTRACTIONS_DIR
    if api_key is None:
        api_key = config.ANTHROPIC_API_KEY

    parsed_dir = Path(parsed_dir)
    output_dir = Path(output_dir)

    # Find all parsed papers — exclude *_classification.json files that may
    # have ended up in the wrong directory.
    json_files = sorted(
        f for f in parsed_dir.glob("*.json")
        if not f.name.endswith("_classification.json")
    )

    if not json_files:
        logger.warning("No parsed paper JSONs found in %s", parsed_dir)
        return []

    # Determine which papers still need classification.
    pending = []
    for jf in json_files:
        paper_id = jf.stem
        cache = output_dir / f"{paper_id}_classification.json"
        if force or not cache.exists():
            pending.append(jf)
        else:
            logger.info("Skipping %s — classification already cached.", paper_id)

    # Always load cached results first so they appear in the final return value.
    cached_results: dict[str, dict] = {}
    for jf in json_files:
        paper_id = jf.stem
        cache = output_dir / f"{paper_id}_classification.json"
        if cache.exists() and paper_id not in [p.stem for p in pending]:
            try:
                cached_results[paper_id] = ClassificationResult.load(cache).to_dict()
            except Exception as exc:
                logger.warning("Could not load cache for %s: %s", paper_id, exc)

    if not pending:
        logger.info("All %d paper(s) already classified.", len(json_files))
        return list(cached_results.values())

    if not api_key:
        logger.error(
            "ANTHROPIC_API_KEY is not set. Cannot classify papers. "
            "Add it to .env or set the environment variable."
        )
        return list(cached_results.values())

    client = anthropic.Anthropic(api_key=api_key)

    logger.info(
        "Classifying %d paper(s) (skipping %d cached).",
        len(pending),
        len(json_files) - len(pending),
    )

    fresh_results: list[dict] = []
    for jf in pending:
        try:
            paper = ParsedPaper.load(jf)
        except Exception as exc:
            logger.error("Cannot load ParsedPaper from %s: %s", jf, exc)
            continue

        result = classify_paper(paper, client, output_dir=output_dir, force=force)
        fresh_results.append(result)

    return list(cached_results.values()) + fresh_results


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def _print_summary_table(results: list[dict]) -> None:
    """Print a formatted summary table of classification results.

    Args:
        results: List of classification result dicts.
    """
    if not results:
        print("No results to display.")
        return

    col_widths = {
        "Paper_ID": max(8, max(len(str(r.get("Paper_ID", ""))) for r in results)),
        "Code": 12,
        "Eligible": 11,
        "Conf": 5,
        "HReview": 8,
        "Flags": 50,
    }

    header = (
        f"{'Paper_ID':<{col_widths['Paper_ID']}}  "
        f"{'Code':<{col_widths['Code']}}  "
        f"{'Eligible':<{col_widths['Eligible']}}  "
        f"{'Conf':<{col_widths['Conf']}}  "
        f"{'HReview':<{col_widths['HReview']}}  "
        f"Flags"
    )
    separator = "-" * (sum(col_widths.values()) + 12)

    print("\n" + "=" * len(separator))
    print("CLASSIFICATION SUMMARY")
    print("=" * len(separator))
    print(header)
    print(separator)

    for r in sorted(results, key=lambda x: x.get("Paper_ID", "")):
        pid = str(r.get("Paper_ID", ""))
        code = str(r.get("Classification_Code", ""))
        eligible = str(r.get("Eligible_Extraction", ""))
        conf = str(r.get("Confidence_Score", ""))
        review = "YES" if r.get("Requires_Human_Review") else "no"
        flags = str(r.get("Flag", ""))[:col_widths["Flags"]]

        print(
            f"{pid:<{col_widths['Paper_ID']}}  "
            f"{code:<{col_widths['Code']}}  "
            f"{eligible:<{col_widths['Eligible']}}  "
            f"{conf:<{col_widths['Conf']}}  "
            f"{review:<{col_widths['HReview']}}  "
            f"{flags}"
        )

    print(separator)

    # Aggregate counts.
    from collections import Counter
    code_counts = Counter(r.get("Classification_Code", "?") for r in results)
    flagged = sum(1 for r in results if r.get("Requires_Human_Review"))
    print(f"\nTotal: {len(results)} paper(s)")
    print("By code: " + ", ".join(f"{k}={v}" for k, v in sorted(code_counts.items())))
    print(f"Requires human review: {flagged}")
    print("=" * len(separator) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify all parsed papers in data/parsed/."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-classify even if a cached result exists.",
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=None,
        help="Override the parsed papers directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the extractions output directory.",
    )
    args = parser.parse_args()

    results = classify_all_papers(
        parsed_dir=args.parsed_dir,
        output_dir=args.output_dir,
        force=args.force,
    )

    if results:
        _print_summary_table(results)
    else:
        print(
            "\nNo papers were classified.\n"
            "  • Ensure data/parsed/ contains at least one {paper_id}.json file\n"
            "    (run python agents/agent_0_parser.py first, or python main.py).\n"
            "  • Ensure ANTHROPIC_API_KEY is set in your .env file.\n"
        )
