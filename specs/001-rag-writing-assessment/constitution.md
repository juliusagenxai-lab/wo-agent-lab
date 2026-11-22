# wo-agent-lab Constitution

## Core Principles

### I. Rubric-First Accuracy (NON-NEGOTIABLE)
Every scoring decision the agent makes MUST be traceable to a rubric criterion retrieved from the knowledge base. The agent MUST NOT score student writing from model memory alone — all scoring rationale must cite retrieved rubric evidence. Hallucinated scores are a critical failure.

### II. Standards Alignment
All rubrics, prompts, and scoring guidance stored in the knowledge base MUST carry explicit CCSS/CA CCSS standard identifiers (e.g., W.6.1). Features that ingest or display rubric content MUST surface those identifiers to users. No rubric content is valid without a standard tag.

### III. Modular Knowledge Base
The knowledge base is grade-band-scoped (Grades 6, 7, 8) and genre-scoped (argumentative, informative/explanatory, narrative). Each document chunk MUST carry metadata: `grade`, `genre`, `standard`, `source`. Retrieval queries MUST filter by at least `grade` + `genre`.

### IV. Transparent Reasoning
Every score returned by the agent MUST include: (a) the rubric criterion name, (b) the score level (1–4), (c) the retrieved descriptor text, and (d) a brief explanation referencing the student's essay. No black-box scores.

### V. Test-First Scoring Validation
Before any scoring pipeline change is deployed, golden-set test cases (student essays with human-assigned scores) MUST be run. A regression that lowers agreement with human scores by more than 5% blocks the release.

### VI. Simplicity Over Abstraction
Start with the simplest retrieval and scoring pipeline that passes the golden-set tests. Add reranking, agentic loops, or multi-hop retrieval only when test evidence justifies the complexity.

## Technology Constraints

- **Language**: Python 3.11+
- **LLM**: GPT-class OSS reasoning model (e.g., via OpenAI-compatible API or Ollama)
- **Embedding model**: Any sentence-transformer or IBM watsonx.ai embedding model
- **Vector store**: Chroma (local/in-memory for dev; persistent for production)
- **Orchestration**: LangChain or LangGraph
- **API layer**: FastAPI (if a web service endpoint is exposed)
- **Grades in scope**: 6, 7, 8 (Grade 6 is the MVP)
- **Genres in scope**: argumentative, informative/explanatory, narrative

## Development Workflow

1. All rubric content is ingested and chunked before any agent code is written.
2. A golden essay set (at least 3 essays per genre × 4 score levels) is created before the scoring pipeline is coded.
3. Scoring pipeline changes are validated against the golden set before merge.
4. Each grade band is a separate vector store namespace/collection.
5. The agent exposes a single `score_essay(grade, genre, essay_text) → ScoringResult` interface; internal retrieval details are hidden from callers.

## Governance

This constitution supersedes all other development conventions for this project. Any architectural decision that violates a principle above requires explicit documentation in a `complexity-justification.md` file citing which principle is violated and why the simpler path is insufficient.

**Version**: 1.0.0 | **Ratified**: 2025-11-22 | **Last Amended**: 2025-11-22
