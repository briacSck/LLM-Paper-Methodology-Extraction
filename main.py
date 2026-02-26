"""
Paper Extraction Pipeline — entry point.

Usage
-----
    python main.py

Runs agent_0_parser on every PDF in data/papers/ and saves JSON to
data/parsed/.  Downstream agents (1–4) will be wired in subsequent phases.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402 — side-effect: loads .env and creates dirs
from agents.agent_0_parser import parse_all_pdfs  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    """Run Phase 1: parse all PDFs in the papers directory."""
    logger.info("=== Phase 1: PDF Parsing ===")
    papers = parse_all_pdfs(
        papers_dir=config.PAPERS_DIR,
        output_dir=config.PARSED_DIR,
    )
    if not papers:
        logger.warning("No papers were parsed. Drop PDFs into %s and re-run.", config.PAPERS_DIR)
        return

    high = sum(1 for p in papers if p.parse_quality == "high")
    medium = sum(1 for p in papers if p.parse_quality == "medium")
    low = sum(1 for p in papers if p.parse_quality == "low")
    logger.info(
        "Parsed %d paper(s) — high: %d | medium: %d | low: %d",
        len(papers),
        high,
        medium,
        low,
    )
    logger.info("JSON files written to %s", config.PARSED_DIR.resolve())


if __name__ == "__main__":
    main()
