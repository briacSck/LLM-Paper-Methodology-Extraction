"""
Schema for a parsed academic paper.

Produced by agent_0_parser and consumed by all downstream agents.
"""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path


@dataclass
class ParsedPaper:
    """Structured representation of a parsed academic PDF."""

    paper_id: str
    title: str
    authors: str
    abstract: str
    theory_hypotheses: str
    methods_data: str
    results: str
    discussion: str
    footnotes: str
    appendix: str
    full_text: str
    page_count: int
    parse_quality: str   # "high" | "medium" | "low"
    parse_notes: str

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialisation."""
        return asdict(self)

    def save(self, directory: Path) -> Path:
        """Serialise to JSON and write to *directory*/{paper_id}.json.

        Args:
            directory: Destination folder (must already exist).

        Returns:
            The path of the written file.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        out_path = directory / f"{self.paper_id}.json"
        out_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return out_path

    @classmethod
    def load(cls, path: Path) -> "ParsedPaper":
        """Load a ParsedPaper from a JSON file produced by :meth:`save`.

        Args:
            path: Path to the JSON file.

        Returns:
            A reconstituted :class:`ParsedPaper` instance.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
