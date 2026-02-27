import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (same directory as this file).
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------

PAPERS_DIR: Path = Path(os.environ.get("PAPERS_DIR", "./data/papers/"))
PARSED_DIR: Path = Path(os.environ.get("PARSED_DIR", "./data/parsed/"))
EXTRACTIONS_DIR: Path = Path(os.environ.get("EXTRACTIONS_DIR", "./data/extractions/"))
OUTPUT_DIR: Path = Path(os.environ.get("OUTPUT_DIR", "./data/output/"))

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

CLAUDE_MODEL: str = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

# ---------------------------------------------------------------------------
# Ensure data directories exist at import time
# ---------------------------------------------------------------------------

for _dir in (PAPERS_DIR, PARSED_DIR, EXTRACTIONS_DIR, OUTPUT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
