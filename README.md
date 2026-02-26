# Paper Extraction Pipeline
### Automated Variable Extraction for Regression-Based Papers
**Missing Data Handling Practices in Management Journals**

---

## Overview

This pipeline automates the classification and structured variable extraction
of 50 academic papers (PDFs) from top management journals (AMJ, ASQ, SMJ, OS,
JOM). It implements a 7-category classification framework to identify
regression-based empirical papers (EQR), then extracts 65 variables across 11
analytical categories (with a primary focus on missing data handling practices).

Built as a 5-phase agentic pipeline using Claude & GPT (Anthropic & OpenAI) for LLM-based
extraction, pdfplumber/PyMuPDF for PDF parsing, and openpyxl for structured
Excel output.

---

## Repository Structure

```
paper-extraction-pipeline/
├── main.py                      # End-to-end pipeline runner
├── config.py                    # Environment and path configuration
├── requirements.txt
├── .env.example
├── agents/
│   ├── agent_0_parser.py        # Phase 1 — PDF parsing + section splitting
│   ├── agent_1_classifier.py    # Phase 2 — 7-category classification (EQR/EQNR-ML/CT/...)
│   ├── agent_2b_extractor.py    # Phase 3 — 65-variable extraction (4 grouped LLM calls)
│   ├── agent_3_qc.py            # Phase 4 — QC rules, auto-correction, human-in-the-loop review queue
│   └── agent_4_exporter.py      # Phase 5 — Excel output + summary report
├── schemas/
│   ├── parsed_paper.py          # ParsedPaper dataclass
│   ├── extraction_schema.py     # 65-variable extraction schema
│   └── qc_schema.py             # QCResult dataclass
├── prompts/                     # LLM system prompts (per agent)
├── data/
│   ├── papers/                  # Drop PDFs here (0001.pdf … 0050.pdf)
│   ├── parsed/                  # (Intermediate Step) JSON per paper
│   ├── extractions/             # Per-paper classification + extraction JSON
│   └── output/                  # Final deliverables
└── tests/
    └── test_parser.py
```

---

## Outputs

| File | Description |
|------|-------------|
| `data/output/extraction_output.xlsx` | 3-sheet Excel: Extraction Data, Extraction Log, Summary Statistics |
| `data/output/human_review_queue.xlsx` | Flagged papers requiring manual review |
| `data/output/summary_report.md` | LLM-generated methods summary |

---

## Classification Framework

Papers are classified into 7 mutually exclusive categories before extraction:

| Code | Category | Eligible for Extraction |
|------|----------|------------------------|
| **EQR** | Empirical Quantitative – Regression | ✅ Yes |
| **EQNR-ML** | ML model performance as primary finding | ❌ No |
| **EQNR-Other** | Quantitative, non-regression, non-ML | ❌ No |
| **MM** | Mixed Methods | ⚠️ Conditional |
| **EQL** | Empirical Qualitative | ❌ No |
| **CT** | Conceptual / Methodological | ❌ No |
| **MA** | Meta-Analysis / Systematic Review | ❌ No |

---

## Setup

### Prerequisites
- Python 3.11+
- Anthropic API key
- OpenAI API key

### Installation

```bash
git clone https://github.com/your-org/paper-extraction-pipeline
cd paper-extraction-pipeline
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

### Add Papers

Drop PDF files into `data/papers/`. Files must follow the naming convention:
`0001.pdf`, `0002.pdf`, …, `0050.pdf`.

---

## Usage

### Run Full Pipeline

```bash
python main.py
```

This executes all 5 phases sequentially: parse → classify → extract → QC → export.

### Run Individual Phases

```bash
python agents/agent_0_parser.py       # Parse all PDFs
python agents/agent_1_classifier.py   # Classify all parsed papers
python agents/agent_2b_extractor.py   # Extract variables (EQR papers only)
python agents/agent_3_qc.py           # Run QC + generate review queue
python agents/agent_4_exporter.py     # Export final Excel + report
```

### Test Parser

```bash
python tests/test_parser.py
```

---

## Extraction Schema (Summary)

65 variables across 11 categories:

1. **Paper Identification** — ID, Authors, Year, Journal, Title
2. **Research Question** — RQ summary, hypotheses count, relationship direction
3. **Dependent Variable** — Name, construct, measurement, source, type
4. **Independent Variable** — Name, construct, measurement, source, type
5. **Mediating Variables** — Presence, method (Baron & Kenny / Bootstrap / SEM…)
6. **Moderating Variables** — Presence, method (interaction term / subgroup…)
7. **Control Variables** — Count, list, justification quality
8. **Sample & Data** — N, context, data type, unit of analysis, time period
9. **Analytical Method** — Model type, endogeneity correction, robustness checks
10. **Missing Data Handling** — Mentioned, rate, method, justification, pattern tests
11. **Replication Potential** — Data/code availability, software, feasibility score

> Category 10 (Missing Data) is the primary focus of this study.

---

## QC & Flags

The pipeline applies 11 automated QC rules and routes flagged papers to a
human-the-loop review queue. Key flags:

| Flag | Trigger |
|------|---------|
| `[FLAG-MULTISTUDY]` | 2+ distinct empirical studies in one paper |
| `[FLAG-ML-UPSTREAM]` | ML constructs IV/DV — verify regression is the hypothesis test |
| `[FLAG-MISSING-AMBIGUOUS]` | Missing data handling inferred, not stated |
| `[FLAG-CLASSIFICATION-UNSTABLE]` | Two LLM classification runs disagree |
| `[FLAG-LPM]` | Binary DV with OLS (linear probability model) |
| `[FLAG-QC-MISSING-LOGIC]` | Missing_Mentioned=0 but sub-fields are not NA |

---

## Environment Variables

```env
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-3-5-sonnet-20241022
PAPERS_DIR=./data/papers/
PARSED_DIR=./data/parsed/
EXTRACTIONS_DIR=./data/extractions/
OUTPUT_DIR=./data/output/
```

---

## Dependencies

```
anthropic>=0.34.0
pdfplumber>=0.11.0
PyMuPDF>=1.24.0
pydantic>=2.0.0
pandas>=2.0.0
openpyxl>=3.1.0
python-dotenv>=1.0.0
```

---

## Pipeline Phases

| Phase | Agent | Input | Output | Verify |
|-------|-------|-------|--------|--------|
| 1 | `agent_0_parser` | PDFs | `parsed/*.json` | `test_parser.py` |
| 2 | `agent_1_classifier` | Parsed JSON | `*_classification.json` | 3+ classifications |
| 3 | `agent_2b_extractor` | Parsed + Classification | `*_extraction.json` | Missing_Handling values |
| 4 | `agent_3_qc` | Extraction JSON | QC results + review queue | `human_review_queue.xlsx` |
| 5 | `agent_4_exporter` | All JSON + reviewed queue | `extraction_output.xlsx` | 3-sheet Excel |

---

*Pipeline developed for academic research purposes. All LLM calls use
temperature=0 for deterministic extraction. Results should be validated against
human coding for a random subsample before use in publication.*
```