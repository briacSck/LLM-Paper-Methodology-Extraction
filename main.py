"""
Paper Extraction Pipeline -- entry point.

Usage
-----
    python main.py

Runs all 5 phases in sequence:
  Phase 1 -- Parse PDFs
  Phase 2 -- Classify papers
  Phase 3 -- Extract structured data (EQR/MM only)
  Phase 4 -- Quality control
  Phase 5 -- Export to Excel + summary report

Cached results from earlier phases are reused automatically so re-runs are fast.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402 -- side-effect: loads .env and creates dirs
from agents.agent_0_parser import parse_all_pdfs  # noqa: E402
from agents.agent_1_classifier import classify_all_papers  # noqa: E402
from agents.agent_2b_extractor import extract_all_papers  # noqa: E402
from agents.agent_3_qc import run_qc_all, generate_review_queue  # noqa: E402
from agents.agent_4_exporter import main as run_export  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main(retry_flags: list[str] | None = None) -> None:
    """Run the full 5-phase extraction pipeline."""

    # ------------------------------------------------------------------
    # Phase 1: Parse PDFs
    # ------------------------------------------------------------------
    logger.info("=== Phase 1: PDF Parsing ===")
    papers = parse_all_pdfs(
        papers_dir=config.PAPERS_DIR,
        output_dir=config.PARSED_DIR,
    )
    if not papers:
        logger.warning(
            "No papers were parsed. Drop PDFs into %s and re-run.", config.PAPERS_DIR
        )
        return

    high   = sum(1 for p in papers if p.parse_quality == "high")
    medium = sum(1 for p in papers if p.parse_quality == "medium")
    low    = sum(1 for p in papers if p.parse_quality == "low")
    logger.info(
        "Parsed %d paper(s) -- high: %d | medium: %d | low: %d",
        len(papers), high, medium, low,
    )

    # ------------------------------------------------------------------
    # Phase 2: Classify
    # ------------------------------------------------------------------
    logger.info("=== Phase 2: Classification ===")
    classify_all_papers(
        parsed_dir=config.PARSED_DIR,
        output_dir=config.EXTRACTIONS_DIR,
        api_key=config.ANTHROPIC_API_KEY,
    )

    # ------------------------------------------------------------------
    # Phase 3: Extract (EQR + MM only)
    # ------------------------------------------------------------------
    logger.info("=== Phase 3: Extraction ===")
    extract_all_papers(
        parsed_dir=config.PARSED_DIR,
        extractions_dir=config.EXTRACTIONS_DIR,
        api_key=config.ANTHROPIC_API_KEY,
        retry_flags=retry_flags,
    )

    # ------------------------------------------------------------------
    # Phase 4: Quality Control
    # ------------------------------------------------------------------
    logger.info("=== Phase 4: Quality Control ===")
    qc_results = run_qc_all(
        extractions_dir=config.EXTRACTIONS_DIR,
        output_dir=config.OUTPUT_DIR,
    )
    generate_review_queue(
        all_qc_results=qc_results,
        extractions_dir=config.EXTRACTIONS_DIR,
        output_dir=config.OUTPUT_DIR,
    )

    # ------------------------------------------------------------------
    # Phase 5: Export
    # ------------------------------------------------------------------
    logger.info("=== Phase 5: Export ===")
    run_export(
        extractions_dir=config.EXTRACTIONS_DIR,
        output_dir=config.OUTPUT_DIR,
        api_key=config.ANTHROPIC_API_KEY,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the full paper extraction pipeline.")
    parser.add_argument(
        "--retry-flags",
        nargs="*",
        default=None,
        help="Only rerun Phase 3 extraction for papers affected by these flags (e.g. FLAG-PARSE-ERROR-b4).",
    )
    args = parser.parse_args()

    main(retry_flags=args.retry_flags)
