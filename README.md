# RAG Writing Assessment & Scoring Agent

A Retrieval-Augmented Generation (RAG) agent that scores Grades 6–8 student essays against CCSS-aligned writing rubrics using a GPT-class OSS reasoning model.

## Overview

The agent retrieves rubric criterion descriptors from a vector store (filtered by grade and genre), then uses an LLM to assign scores (1–4) for each of six rubric criteria, with transparent rationale citing the retrieved rubric text.

**Supported grades**: 6, 7, 8  
**Supported genres**: argumentative, informative, narrative  
**Standards alignment**: CCSS / CA CCSS Writing Standards W.6–8.1–3

---

## Project Structure

```
src/
├── models/        # Pydantic data models
├── ingestion/     # Rubric loading, chunking, vector store ingestion
├── retrieval/     # Chroma retriever (grade + genre filtered)
├── scoring/       # Prompt templates, scorer, validator
└── api/           # FastAPI endpoint (Phase 3+)

data/rubrics/      # Rubric Markdown source files (grade_6/, grade_7/, grade_8/)
tests/
├── golden/        # Human-scored golden essays for validation
├── unit/          # Unit tests
└── integration/   # Integration tests (golden set scoring)

specs/001-rag-writing-assessment/   # SDD artifacts
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your LLM endpoint and model name
```

### 3. Ingest rubrics into the vector store

```bash
python -m src.ingestion.ingest
```

This loads all rubric Markdown files from `data/rubrics/`, chunks them by criterion, embeds them, and stores them in Chroma collections (`rubrics_grade_6`, `rubrics_grade_7`, `rubrics_grade_8`).

---

## Usage

### CLI — score a single essay

```bash
python -m src.cli --grade 6 --genre argumentative --file path/to/essay.txt
```

### CLI — score all three genres for one student

```bash
python -m src.cli --all --grade 6 \
  --argumentative arg_essay.txt \
  --informative inf_essay.txt \
  --narrative nar_essay.txt
```

### Python API

```python
from src.scoring.scorer import score_essay

result = score_essay(grade=6, genre="argumentative", essay_text="...")
print(result.model_dump_json(indent=2))
```

---

## Scoring Output

Each `AssessmentScore` contains:
- Six `CriterionScore` objects, each with:
  - `criterion_name` — e.g., "Claim"
  - `level` — 1 (Beginning) through 4 (Advanced)
  - `descriptor` — the retrieved rubric text for that level
  - `rationale` — one sentence referencing the student's essay
- `total` — sum of six criteria (max 24)
- `performance_level` — Advanced / Proficient / Developing / Beginning
- `standard` — CCSS standard identifier (e.g., W.6.1)

---

## Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (requires ingested vector store + running LLM)
pytest tests/integration/ -v

# Full suite with coverage
pytest --cov=src --cov-report=term-missing
```

---

## Architecture

```
essay_text + grade + genre
        │
        ▼
  [Validator]  ← rejects bad inputs immediately
        │
        ▼
  [Retriever]  ← Chroma, filtered by grade + genre
        │        returns 6 criterion descriptor chunks
        ▼
  [Prompt Builder]  ← injects rubric descriptors + essay
        │
        ▼
  [LLM (OSS)]  ← returns structured JSON scores
        │
        ▼
  [Output Parser + Clamper]  ← validates 1–4, retries on bad JSON
        │
        ▼
  AssessmentScore
```

---

## Spec-Driven Development Artifacts

See [`specs/001-rag-writing-assessment/`](specs/001-rag-writing-assessment/) for:
- `constitution.md` — project principles
- `spec.md` — user stories and requirements
- `plan.md` — implementation plan and tech context
- `tasks.md` — full task breakdown
