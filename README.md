# LLM-Paper-Methodology-Extraction
### Automated Variable Extraction for Regression-Based Papers
**Missing Data Handling Practices in Management Journals** — Oxford RA Screening Task

---

## Overview

This pipeline automates the classification and structured variable extraction
of 50 academic papers (PDFs) from top management journals (AMJ, ASQ, SMJ, OS,
JOM). It implements a 7-category classification framework to identify
regression-based empirical papers (EQR), then extracts 65 variables across 11
analytical categories (with a primary focus on missing data handling practices).

Built as a 5-phase agentic pipeline using **Claude (Anthropic)** for LLM-based extraction, **pdfplumber/PyMuPDF** for PDF parsing, and **openpyxl** for structured
Excel output.

---

## Results — 50-Paper Sample

Pipeline executed on the full 50-paper sample (February 2026, SMJ corpus).
All statistics below are drawn from `data/output/extraction_output.xlsx`.

---

### Classification Breakdown

| Category | Label | Count | % of Sample |
|----------|-------|-------|-------------|
| **EQR** | Empirical Quantitative – Regression | **27** | **54.0%** |
| CT | Conceptual / Theoretical | 10 | 20.0% |
| MM | Mixed Methods | 7 | 14.0% |
| EQL | Empirical Qualitative | 2 | 4.0% |
| EQNR-Other | Quantitative, non-regression | 2 | 4.0% |
| EQNR-ML | ML-focused | 1 | 2.0% |
| MA | Meta-Analysis | 1 | 2.0% |

→ **27 papers eligible for full variable extraction (EQR).**

---

### Missing Data Handling (EQR papers, n = 27)

| Dimension | n | % of EQR |
|-----------|---|----------|
| Missing data mentioned in any form | 10 | 37.0% |
| Missing data **not** mentioned | 13 | 48.1% |
| Missing rate explicitly reported | 1 | 3.7% |
| Method justified by authors | 1 | 3.7% |
| Missing pattern tested (MCAR/MAR/MNAR) | **0** | **0.0%** |
| Sensitivity analysis on missing data | 1 | 3.7% |

**Handling methods identified:**

| Method | Count | % of EQR |
|--------|-------|----------|
| Unknown / not reported | 17 | 63.0% |
| Listwise deletion | 6 | 22.2% |
| Other | 4 | 14.8% |
| Multiple imputation (MI) | 0 | 0.0% |
| Full information maximum likelihood (FIML) | 0 | 0.0% |

> Key finding: **63% of EQR papers provide no identifiable missing data handling method.
> 0% test for missing data patterns. Neither MI nor FIML was used in any paper.**

---

### Model Types (EQR papers, n = 27)

| Model | Count | % of EQR |
|-------|-------|----------|
| Tobit / fixed-effects panel | 16 | 59.3% |
| Unknown | 4 | 14.8% |
| Survival analysis | 2 | 7.4% |
| Factor analysis / SEM | 2 | 7.4% |
| OLS / Linear | 2 | 7.4% |
| Logit / Probit | 1 | 3.7% |

---

### Reporting Transparency (EQR papers, n = 27)

| Indicator | Yes | % | No | % |
|-----------|-----|---|----|---|
| Software reported | 7 | 25.9% | 16 | 59.3% |
| Code available | 5 | 18.5% | 18 | 66.7% |
| Replication feasible | 3 | 11.1% | — | — |

---

### Pipeline Performance

- **Papers processed**: 50 / 50
- **Auto-classification success**: 100%
- **Full extraction completed**: 27 EQR papers
- **Human review queue**: papers flagged with `FLAG-ML-UPSTREAM`, `FLAG-MULTISTUDY`, `FLAG-MISSING-AMBIGUOUS`, `FLAG-CLASSIFICATION-UNSTABLE`

---

## Repository Structure

```
LLM-Paper-Methodology-Extraction/
├── main.py                      # End-to-end pipeline runner
├── config.py                    # Environment and path configuration
├── requirements.txt
├── .gitignore
├── agents/
│   ├── agent_0_parser.py        # Phase 1 — PDF parsing + section splitting
│   ├── agent_1_classifier.py    # Phase 2 — 7-category classification
│   ├── agent_2b_extractor.py    # Phase 3 — 65-variable extraction (4 grouped LLM calls)
│   ├── agent_3_qc.py            # Phase 4 — QC rules, auto-correction, human-in-the-loop
│   └── agent_4_exporter.py      # Phase 5 — Excel output + summary report
├── schemas/
│   ├── parsed_paper.py          # ParsedPaper dataclass
│   ├── extraction_schema.py     # 65-variable extraction schema + QC flag injection
│   └── qc_schema.py             # QCResult dataclass
├── data/
│   ├── papers/                  # Drop PDFs here (0001.pdf … 0050.pdf) — not tracked
│   ├── parsed/                  # Intermediate JSON per paper — not tracked
│   ├── extractions/             # Per-paper classification + extraction JSON — not tracked
│   └── output/                  # Final deliverables — not tracked
└── tests/
    ├── test_parser.py           # Integration test — PDF parsing (requires real PDFs)
    ├── test_classifier.py       # Unit tests — classify_paper() (fully offline, mocked)
    ├── test_extractor.py        # Unit tests — ExtractionResult schema (no LLM calls)
    └── fixtures/
        └── mock_paper.json      # Fictional parsed paper for offline testing
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
- Anthropic API (`ANTHROPIC_API_KEY`)

### Installation

```bash
git clone https://github.com/briacSck/LLM-Paper-Methodology-Extraction
cd LLM-Paper-Methodology-Extraction
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Create a `.env` file at the root:

```text
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-3-5-sonnet-20241022
PAPERS_DIR=./data/papers/
PARSED_DIR=./data/parsed/
EXTRACTIONS_DIR=./data/extractions/
OUTPUT_DIR=./data/output/
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

## Pipeline Phases

| Phase | Agent | Input | Output | Verify |
|-------|-------|-------|--------|--------|
| 1 | `agent_0_parser` | PDFs | `parsed/*.json` | `test_parser.py` |
| 2 | `agent_1_classifier` | Parsed JSON | `*_classification.json` | 3+ classifications |
| 3 | `agent_2b_extractor` | Parsed + Classification | `*_extraction.json` | Missing_Handling values |
| 4 | `agent_3_qc` | Extraction JSON | QC results + review queue | `human_review_queue.xlsx` |
| 5 | `agent_4_exporter` | All JSON + reviewed queue | `extraction_output.xlsx` | 3-sheet Excel |

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
| `FLAG-MULTISTUDY` | 2+ distinct empirical studies in one paper |
| `FLAG-ML-UPSTREAM` | ML constructs IV/DV — verify regression is the hypothesis test |
| `FLAG-MISSING-AMBIGUOUS` | Missing data handling inferred, not stated |
| `FLAG-CLASSIFICATION-UNSTABLE` | Two LLM classification runs disagree |
| `FLAG-LPM` | Binary DV with OLS (linear probability model) |
| `FLAG-QC-MISSING-LOGIC` | Missing_Mentioned=0 but sub-fields are not NA |

---

## Rate Limiting & Reliability

The extraction agent (`agent_2b_extractor.py`) includes:
+ Retry logic with exponential backoff on Anthropic API rate-limit errors
+ Cache protection — rate-limited extractions are not written to disk, preventing corrupt cached results from blocking reruns
+ Resumable runs — already-extracted papers are skipped automatically on re-execution

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

