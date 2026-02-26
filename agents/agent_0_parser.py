"""
Agent 0 — PDF Parser.

Responsibilities
----------------
* Extract raw text from a PDF using pdfplumber (primary) or PyMuPDF (fallback).
* Tag each page: ``--- PAGE N ---\\n<text>``.
* Split the concatenated text into named academic sections via regex heuristics.
* Attempt lightweight title / author extraction from the first page.
* Persist the result as a :class:`~schemas.parsed_paper.ParsedPaper` JSON file.

No LLM calls are made in this module — pure text processing only.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Allow running the module directly from inside the agents/ folder.
_AGENTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _AGENTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402 — loads .env
from schemas.parsed_paper import ParsedPaper  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Section-detection regex
# ---------------------------------------------------------------------------
# Each entry is (section_key, compiled_pattern).
# Patterns are matched against individual lines (case-insensitive).
# The *first* match in the text determines where that section starts.

_SECTION_PATTERNS: list[Tuple[str, re.Pattern]] = [
    (
        "abstract",
        re.compile(
            r"^\s*abstract\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "introduction",
        re.compile(
            r"^\s*(?:\d+[\.\s]+)?introduction\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "theory_hypotheses",
        re.compile(
            r"^\s*(?:\d+[\.\s]+)?(?:theory|theoretical\s+(?:background|framework)|"
            r"hypothes[ie]s|literature\s+review|conceptual\s+framework|"
            r"background\s+and\s+hypothes[ie]s|prior\s+research|related\s+literature)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "methods_data",
        re.compile(
            r"^\s*(?:\d+[\.\s]+)?(?:data|method(?:s|ology)?|empirical\s+(?:strategy|analysis)|"
            r"research\s+design|sample|measures?|variables?|model|estimation|"
            r"data\s+and\s+methods?|methods?\s+and\s+data)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "results",
        re.compile(
            r"^\s*(?:\d+[\.\s]+)?(?:results?|findings?|empirical\s+results?|"
            r"analyses?|main\s+results?|analysis)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "discussion",
        re.compile(
            r"^\s*(?:\d+[\.\s]+)?(?:discussion|conclusion(?:s)?|"
            r"discussion\s+and\s+conclusion(?:s)?|concluding\s+remarks?|"
            r"implications?|general\s+discussion)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "footnotes",
        re.compile(
            r"^\s*(?:foot\s*notes?|end\s*notes?|notes?)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "appendix",
        re.compile(
            r"^\s*(?:appendix|appendices|online\s+(?:supplement|appendix)|"
            r"supplemental\s+(?:material|appendix)|web\s+appendix)\s*[A-Z]?\s*$",
            re.IGNORECASE,
        ),
    ),
]

# Ordered list used for quality assessment (core sections that should be present).
_CORE_SECTIONS = ["abstract", "methods_data", "results"]


# ---------------------------------------------------------------------------
# Low-level text extraction helpers
# ---------------------------------------------------------------------------


def _extract_with_pdfplumber(pdf_path: Path) -> Tuple[str, int]:
    """Extract full text using pdfplumber, page by page.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        Tuple of (full_text_with_page_markers, page_count).

    Raises:
        Exception: Re-raises any pdfplumber error so the caller can fall back.
    """
    import pdfplumber  # lazy import — keeps module importable without the package

    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(f"--- PAGE {i} ---\n{text.strip()}")
    return "\n\n".join(pages), page_count


def _extract_with_pymupdf(pdf_path: Path) -> Tuple[str, int]:
    """Extract full text using PyMuPDF (fitz), page by page.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        Tuple of (full_text_with_page_markers, page_count).

    Raises:
        Exception: Re-raises any fitz error so the caller can propagate it.
    """
    import fitz  # PyMuPDF — lazy import

    pages: list[str] = []
    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        pages.append(f"--- PAGE {i} ---\n{text.strip()}")
    doc.close()
    return "\n\n".join(pages), page_count


def _extract_text(pdf_path: Path) -> Tuple[str, int, str]:
    """Try pdfplumber first; fall back to PyMuPDF on any error.

    Args:
        pdf_path: Path to the PDF.

    Returns:
        Tuple of (full_text, page_count, extraction_method) where
        extraction_method is ``"pdfplumber"`` or ``"pymupdf"``.

    Raises:
        RuntimeError: If both extractors fail.
    """
    # Capture in a plain variable so it remains accessible after the except
    # block ends (Python 3.11+ deletes the ``as`` binding on block exit).
    primary_exc_str: str = ""
    try:
        full_text, page_count = _extract_with_pdfplumber(pdf_path)
        logger.info("pdfplumber succeeded for %s (%d pages)", pdf_path.name, page_count)
        return full_text, page_count, "pdfplumber"
    except Exception as exc:
        primary_exc_str = str(exc)
        logger.warning(
            "pdfplumber failed for %s (%s); trying PyMuPDF fallback.",
            pdf_path.name,
            exc,
        )

    try:
        full_text, page_count = _extract_with_pymupdf(pdf_path)
        logger.info(
            "PyMuPDF fallback succeeded for %s (%d pages)", pdf_path.name, page_count
        )
        return full_text, page_count, "pymupdf"
    except Exception as exc:
        raise RuntimeError(
            f"Both pdfplumber and PyMuPDF failed for {pdf_path.name}. "
            f"Primary: {primary_exc_str}. Fallback: {exc}."
        ) from exc


# ---------------------------------------------------------------------------
# Title / author extraction
# ---------------------------------------------------------------------------


def _extract_title_and_authors(first_page_text: str) -> Tuple[str, str]:
    """Heuristically extract title and author line from the first page.

    The title is assumed to be the longest non-trivial line in the first 20
    lines (or up to the first blank line after the opening block).  Authors
    are assumed to follow the title on the next non-blank line(s).

    Args:
        first_page_text: Raw text of the first page (no page marker).

    Returns:
        Tuple of (title, authors).  Either may be an empty string if not found.
    """
    lines = [ln.strip() for ln in first_page_text.splitlines() if ln.strip()]
    if not lines:
        return "", ""

    # Use the first 25 lines to avoid going deep into the abstract.
    candidates = lines[:25]

    # Title heuristic: longest line that is not a journal/volume/doi line.
    _skip_patterns = re.compile(
        r"^\s*(?:doi|http|volume|vol\.|issue|journal|©|copyright|\d{4})",
        re.IGNORECASE,
    )
    title_candidates = [
        ln for ln in candidates if not _skip_patterns.match(ln) and len(ln) > 10
    ]
    title = max(title_candidates, key=len) if title_candidates else candidates[0]

    # Authors heuristic: the line immediately after the title (if present).
    title_idx = candidates.index(title) if title in candidates else 0
    author_lines: list[str] = []
    for ln in candidates[title_idx + 1 : title_idx + 4]:
        # Stop if we hit what looks like an institutional affiliation or abstract.
        if re.match(r"abstract|received|accepted|available", ln, re.IGNORECASE):
            break
        author_lines.append(ln)
    authors = " | ".join(author_lines)

    return title, authors


# ---------------------------------------------------------------------------
# Section splitter
# ---------------------------------------------------------------------------


def _split_into_sections(full_text: str) -> Dict[str, str]:
    """Detect academic section boundaries and return a section→text mapping.

    Uses the ordered patterns in :data:`_SECTION_PATTERNS`.  For each pattern
    the *first* matching line in the text is treated as the section header.
    Text between consecutive headers is assigned to the earlier section.

    Args:
        full_text: The entire paper text (with ``--- PAGE N ---`` markers).

    Returns:
        Dict mapping section keys to their extracted text.  Keys that were not
        found are present but mapped to an empty string.

    Notes:
        Returns ``None`` in the ``_found`` meta-key if no sections were found.
    """
    lines = full_text.splitlines()
    n = len(lines)

    # Map section_key → line index of its header.
    header_positions: Dict[str, int] = {}
    for section_key, pattern in _SECTION_PATTERNS:
        for idx, line in enumerate(lines):
            if pattern.match(line):
                header_positions[section_key] = idx
                break  # take the first occurrence only

    if not header_positions:
        return {key: "" for key, _ in _SECTION_PATTERNS}

    # Sort found sections by their position in the document.
    ordered = sorted(header_positions.items(), key=lambda kv: kv[1])

    sections: Dict[str, str] = {key: "" for key, _ in _SECTION_PATTERNS}
    for i, (key, start_line) in enumerate(ordered):
        # The section text runs from the line AFTER the header to just before
        # the next detected section header (or the end of the document).
        end_line = ordered[i + 1][1] if i + 1 < len(ordered) else n
        section_text = "\n".join(lines[start_line + 1 : end_line]).strip()
        sections[key] = section_text

    return sections


# ---------------------------------------------------------------------------
# Quality assessment
# ---------------------------------------------------------------------------


def _assess_quality(sections: Dict[str, str], extraction_method: str) -> Tuple[str, str]:
    """Return a (parse_quality, parse_notes) pair.

    Quality rubric
    ~~~~~~~~~~~~~~
    * **high**   — all three core sections (abstract, methods_data, results)
                   are non-empty and each is at least 100 characters long.
    * **medium** — at least one core section found or extraction fell back to
                   PyMuPDF.
    * **low**    — no core sections found (or fallback set externally).

    Args:
        sections: Dict from :func:`_split_into_sections`.
        extraction_method: ``"pdfplumber"`` or ``"pymupdf"``.

    Returns:
        Tuple of (quality_label, human_readable_notes).
    """
    found_core = [s for s in _CORE_SECTIONS if len(sections.get(s, "")) >= 100]
    notes_parts: list[str] = [f"Extracted via {extraction_method}."]

    if len(found_core) == len(_CORE_SECTIONS):
        quality = "high"
    elif found_core:
        quality = "medium"
        missing = [s for s in _CORE_SECTIONS if s not in found_core]
        notes_parts.append(f"Missing or short core sections: {missing}.")
    else:
        quality = "low"
        notes_parts.append("No core sections detected; full_text preserved.")

    if extraction_method == "pymupdf":
        if quality == "high":
            quality = "medium"
        notes_parts.append("pdfplumber failed; used PyMuPDF fallback.")

    return quality, " ".join(notes_parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_pdf(pdf_path: Path | str, output_dir: Optional[Path] = None) -> ParsedPaper:
    """Parse a single PDF and return a :class:`~schemas.parsed_paper.ParsedPaper`.

    This function **never raises** — all exceptions are caught, logged, and
    surfaced through ``parse_quality="low"`` and ``parse_notes``.

    Args:
        pdf_path: Path to the PDF file to parse.
        output_dir: Directory in which to save the JSON output.  Defaults to
            ``config.PARSED_DIR``.

    Returns:
        A populated :class:`~schemas.parsed_paper.ParsedPaper` instance.  The
        JSON is also written to *output_dir*/{paper_id}.json.
    """
    pdf_path = Path(pdf_path).resolve()
    if output_dir is None:
        output_dir = config.PARSED_DIR

    paper_id = pdf_path.stem  # filename without extension

    # ------------------------------------------------------------------
    # Minimal "error" paper returned on catastrophic failure.
    # ------------------------------------------------------------------
    def _error_paper(note: str) -> ParsedPaper:
        paper = ParsedPaper(
            paper_id=paper_id,
            title="",
            authors="",
            abstract="",
            theory_hypotheses="",
            methods_data="",
            results="",
            discussion="",
            footnotes="",
            appendix="",
            full_text="",
            page_count=0,
            parse_quality="low",
            parse_notes=note,
        )
        try:
            paper.save(output_dir)
        except Exception as save_exc:
            logger.error("Could not save error paper JSON: %s", save_exc)
        return paper

    # ------------------------------------------------------------------
    # Step 1: Extract raw text
    # ------------------------------------------------------------------
    try:
        full_text, page_count, extraction_method = _extract_text(pdf_path)
    except Exception as exc:
        msg = f"Text extraction failed: {exc}"
        logger.error(msg)
        return _error_paper(msg)

    if not full_text.strip():
        msg = "Extracted text is empty (scanned/image-only PDF?)."
        logger.warning(msg)
        return _error_paper(msg)

    # ------------------------------------------------------------------
    # Step 2: Title and authors from first page
    # ------------------------------------------------------------------
    try:
        first_page_text = full_text.split("--- PAGE 2 ---")[0]
        # Strip the page marker itself.
        first_page_text = re.sub(r"--- PAGE \d+ ---", "", first_page_text).strip()
        title, authors = _extract_title_and_authors(first_page_text)
    except Exception as exc:
        logger.warning("Title/author extraction failed: %s", exc)
        title, authors = "", ""

    # ------------------------------------------------------------------
    # Step 3: Section splitting
    # ------------------------------------------------------------------
    try:
        sections = _split_into_sections(full_text)
    except Exception as exc:
        logger.warning("Section splitting failed: %s", exc)
        sections = {key: "" for key, _ in _SECTION_PATTERNS}

    # ------------------------------------------------------------------
    # Step 4: Quality assessment
    # ------------------------------------------------------------------
    try:
        quality, notes = _assess_quality(sections, extraction_method)
    except Exception as exc:
        logger.warning("Quality assessment failed: %s", exc)
        quality, notes = "low", f"Quality assessment error: {exc}"

    # ------------------------------------------------------------------
    # Step 5: Assemble and persist
    # ------------------------------------------------------------------
    paper = ParsedPaper(
        paper_id=paper_id,
        title=title,
        authors=authors,
        abstract=sections.get("abstract", ""),
        theory_hypotheses=sections.get("theory_hypotheses", ""),
        methods_data=sections.get("methods_data", ""),
        results=sections.get("results", ""),
        discussion=sections.get("discussion", ""),
        footnotes=sections.get("footnotes", ""),
        appendix=sections.get("appendix", ""),
        full_text=full_text,
        page_count=page_count,
        parse_quality=quality,
        parse_notes=notes,
    )

    try:
        out_path = paper.save(output_dir)
        logger.info("Saved parsed paper to %s", out_path)
    except Exception as exc:
        logger.error("Failed to save parsed paper: %s", exc)

    return paper


def parse_all_pdfs(
    papers_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> list[ParsedPaper]:
    """Parse every PDF in *papers_dir* and return all :class:`ParsedPaper` objects.

    Skips PDFs that already have a cached JSON in *output_dir*.

    Args:
        papers_dir: Folder containing PDF files.  Defaults to
            ``config.PAPERS_DIR``.
        output_dir: Destination for JSON files.  Defaults to
            ``config.PARSED_DIR``.

    Returns:
        List of :class:`ParsedPaper` instances (one per PDF found).
    """
    if papers_dir is None:
        papers_dir = config.PAPERS_DIR
    if output_dir is None:
        output_dir = config.PARSED_DIR
    papers_dir = Path(papers_dir)
    output_dir = Path(output_dir)

    pdf_files = sorted(papers_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", papers_dir)
        return []

    logger.info("Found %d PDF(s) in %s", len(pdf_files), papers_dir)
    results: list[ParsedPaper] = []
    for pdf in pdf_files:
        paper_id = pdf.stem
        cache_path = output_dir / f"{paper_id}.json"
        if cache_path.exists():
            logger.info("Cache hit for %s — skipping re-parse.", paper_id)
            try:
                results.append(ParsedPaper.load(cache_path))
                continue
            except Exception as exc:
                logger.warning(
                    "Failed to load cache for %s (%s); re-parsing.", paper_id, exc
                )
        logger.info("Parsing %s …", pdf.name)
        results.append(parse_pdf(pdf, output_dir=output_dir))

    return results

