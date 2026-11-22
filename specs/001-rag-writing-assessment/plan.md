# Implementation Plan: RAG Writing Assessment & Scoring Agent

**Branch**: `001-rag-writing-assessment` | **Date**: 2025-11-22 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-rag-writing-assessment/spec.md`

---

## Summary

Build a Retrieval-Augmented Generation (RAG) agent that scores Grades 6–8 student essays against CCSS-aligned writing rubrics. The agent retrieves rubric criterion descriptors from a vector store (filtered by grade and genre), then uses a GPT-class OSS reasoning model to assign scores (1–4) for each of six rubric criteria, with transparent rationale. Grade 6 is the MVP; Grades 7–8 extend the same pipeline. The public interface is `score_essay(grade, genre, essay_text) → ScoringResult`.

---

## Technical Context

**Language/Version**: Python 3.11+

**Primary Dependencies**:
- `langchain` / `langchain-community` — RAG pipeline orchestration
- `chromadb` — vector store (persistent, grade-band namespaced)
- `sentence-transformers` or `ibm-watsonx-ai` — embedding model
- `openai` (OpenAI-compatible client) — LLM inference (OSS model via Ollama or remote endpoint)
- `pydantic` v2 — data models and validation
- `fastapi` + `uvicorn` — optional HTTP API layer (Phase 3+)
- `pytest` — test framework
- `python-dotenv` — environment variable management

**Storage**: Chroma persistent vector store; one collection per grade band (`rubrics_grade_6`, `rubrics_grade_7`, `rubrics_grade_8`)

**Testing**: `pytest`; golden essay set in `tests/golden/`

**Target Platform**: Local development (Windows/Mac/Linux); deployable as a Python service

**Project Type**: Python library + CLI + optional web service

**Performance Goals**: End-to-end `score_essay` ≤ 15 seconds for a 500-word essay

**Constraints**: No scoring from model memory — retrieval MUST precede every LLM call; all rubric sources must carry CCSS standard metadata

**Scale/Scope**: MVP Grade 6 (3 genres × 6 criteria = 18 rubric descriptor sets); full scope Grades 6–8 (54 rubric descriptor sets)

---

## Constitution Check

| Principle | Status | Notes |
|---|---|---|
| I. Rubric-First Accuracy | ✅ PASS | Retrieval gate enforced before every LLM call |
| II. Standards Alignment | ✅ PASS | Metadata schema includes `standard` field on every chunk |
| III. Modular Knowledge Base | ✅ PASS | Separate Chroma collections per grade; metadata filters on every query |
| IV. Transparent Reasoning | ✅ PASS | ScoringResult exposes criterion name, level, descriptor, rationale |
| V. Test-First Scoring Validation | ✅ PASS | Golden essay set created before pipeline coding (Phase 1) |
| VI. Simplicity Over Abstraction | ✅ PASS | LangChain retrieval chain, no agentic loops in MVP |

---

## Project Structure

### Documentation (this feature)

```text
specs/001-rag-writing-assessment/
├── constitution.md      # Project principles
├── spec.md              # Feature specification (this doc's input)
├── plan.md              # This file
├── research.md          # Rubric ingestion and embedding strategy notes
├── data-model.md        # Pydantic model definitions
├── quickstart.md        # How to run ingestion + score an essay
└── tasks.md             # Phase 4 output — actionable task list
```

### Source Code (repository root)

```text
src/
├── models/
│   ├── essay.py             # Essay input model (grade, genre, text)
│   ├── rubric.py            # RubricCriterion, CriterionScore
│   └── result.py            # AssessmentScore, StudentReport
├── ingestion/
│   ├── loader.py            # Load rubric Markdown files
│   ├── chunker.py           # Chunk rubric content with metadata
│   └── ingest.py            # CLI entry point: python -m src.ingestion.ingest
├── retrieval/
│   ├── store.py             # Chroma client, collection management
│   └── retriever.py         # Filtered retrieval by grade + genre
├── scoring/
│   ├── prompt.py            # Prompt templates for scoring LLM
│   ├── scorer.py            # score_essay() — main public interface
│   └── validator.py         # Input validation, output clamping
├── api/
│   └── main.py              # FastAPI app (Phase 3+)
└── cli.py                   # CLI wrapper for score_essay

data/
└── rubrics/
    ├── grade_6/
    │   ├── argumentative.md
    │   ├── informative.md
    │   └── narrative.md
    ├── grade_7/              # Populated before Phase 4 (US4)
    └── grade_8/              # Populated before Phase 4 (US4)

tests/
├── golden/
│   ├── grade_6_argumentative/ # Essays + human scores (P1, P2, P3, P4 levels)
│   ├── grade_6_informative/
│   └── grade_6_narrative/
├── unit/
│   ├── test_models.py
│   ├── test_validator.py
│   └── test_scoring.py
└── integration/
    └── test_score_essay.py
```

**Structure Decision**: Single project (Option 1). Library + CLI first; API layer added in later phase.

---

## Complexity Tracking

> No constitution violations requiring justification.
