"""
Tests for agent_0_parser.

Run with:
    python tests/test_parser.py

Or with pytest:
    pytest tests/test_parser.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402 — loads .env
from agents.agent_0_parser import parse_pdf  # noqa: E402
from schemas.parsed_paper import ParsedPaper  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_first_pdf() -> Path | None:
    """Return the first PDF found in config.PAPERS_DIR, or None."""
    pdfs = sorted(config.PAPERS_DIR.glob("*.pdf"))
    return pdfs[0] if pdfs else None


def _print_section_summary(paper: ParsedPaper) -> None:
    """Print a summary of section lengths and quality to stdout."""
    print(f"\n{'=' * 60}")
    print(f"Paper ID   : {paper.paper_id}")
    print(f"Title      : {paper.title[:80]!r}")
    print(f"Authors    : {paper.authors[:80]!r}")
    print(f"Pages      : {paper.page_count}")
    print(f"Quality    : {paper.parse_quality}")
    print(f"Notes      : {paper.parse_notes}")
    print(f"\nSection lengths (chars):")
    sections = {
        "abstract": paper.abstract,
        "theory_hypotheses": paper.theory_hypotheses,
        "methods_data": paper.methods_data,
        "results": paper.results,
        "discussion": paper.discussion,
        "footnotes": paper.footnotes,
        "appendix": paper.appendix,
    }
    for name, text in sections.items():
        print(f"  {name:<22}: {len(text):>6}")
    print(f"  {'full_text':<22}: {len(paper.full_text):>6}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_import() -> None:
    """Verify that the parser module imports without error."""
    from agents.agent_0_parser import parse_pdf, parse_all_pdfs  # noqa: F401
    print("PASS  test_import")


def test_parse_single_pdf() -> None:
    """Parse the first available PDF and validate the returned object."""
    pdf = _find_first_pdf()
    if pdf is None:
        print(
            "SKIP  test_parse_single_pdf — no PDF files found in "
            f"{config.PAPERS_DIR.resolve()}\n"
            "      Drop at least one PDF into that folder and re-run."
        )
        return

    print(f"\nParsing: {pdf.name}")
    paper = parse_pdf(pdf)

    # Basic type check.
    assert isinstance(paper, ParsedPaper), "parse_pdf must return a ParsedPaper"

    # paper_id must equal the stem of the file.
    assert paper.paper_id == pdf.stem, (
        f"Expected paper_id={pdf.stem!r}, got {paper.paper_id!r}"
    )

    # full_text must be non-empty (even scanned PDFs should produce something).
    assert paper.full_text or paper.parse_quality == "low", (
        "full_text is empty but parse_quality is not 'low'"
    )

    # parse_quality must be one of the three valid values.
    assert paper.parse_quality in {"high", "medium", "low"}, (
        f"Invalid parse_quality: {paper.parse_quality!r}"
    )

    # JSON output must have been written to disk.
    expected_json = config.PARSED_DIR / f"{paper.paper_id}.json"
    assert expected_json.exists(), f"Expected JSON at {expected_json} but file not found"

    _print_section_summary(paper)
    print("PASS  test_parse_single_pdf")


def test_methods_data_not_empty() -> None:
    """Assert that methods_data is non-empty for a successfully parsed paper.

    This test is skipped if no PDF is available or if the paper quality is
    'low' (indicating the parser could not detect sections at all).
    """
    pdf = _find_first_pdf()
    if pdf is None:
        print(
            "SKIP  test_methods_data_not_empty — no PDF in "
            f"{config.PAPERS_DIR.resolve()}"
        )
        return

    paper = parse_pdf(pdf)

    if paper.parse_quality == "low":
        print(
            f"SKIP  test_methods_data_not_empty — parse_quality='low' "
            f"(notes: {paper.parse_notes})"
        )
        return

    assert paper.methods_data, (
        "methods_data is empty for a paper with parse_quality != 'low'.\n"
        f"  parse_notes: {paper.parse_notes}\n"
        "  Check that the PDF contains a recognisable methods/data section header."
    )
    print(f"PASS  test_methods_data_not_empty  (methods_data: {len(paper.methods_data)} chars)")


def test_error_recovery() -> None:
    """Confirm that parse_pdf returns a low-quality paper for a non-existent file."""
    fake_path = config.PAPERS_DIR / "__does_not_exist__.pdf"
    paper = parse_pdf(fake_path)
    assert paper.parse_quality == "low", (
        "Expected parse_quality='low' for a missing file"
    )
    assert paper.parse_notes, "parse_notes must describe the error"
    print(f"PASS  test_error_recovery  (notes: {paper.parse_notes[:80]})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    failures: list[str] = []

    for test_fn in [
        test_import,
        test_error_recovery,
        test_parse_single_pdf,
        test_methods_data_not_empty,
    ]:
        try:
            test_fn()
        except AssertionError as exc:
            failures.append(f"FAIL  {test_fn.__name__}: {exc}")
            print(failures[-1])
        except Exception as exc:
            failures.append(f"ERROR {test_fn.__name__}: {exc}")
            print(failures[-1])

    print()
    if failures:
        print(f"{len(failures)} test(s) FAILED.")
        sys.exit(1)
    else:
        print("All tests passed.")
