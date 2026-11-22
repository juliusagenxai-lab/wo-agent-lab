---
description: "Task list for RAG Writing Assessment & Scoring Agent"
---

# Tasks: RAG Writing Assessment & Scoring Agent

**Input**: Design documents from `specs/001-rag-writing-assessment/`

**Prerequisites**: plan.md ✅ | spec.md ✅

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, dependencies, and golden essay set — MUST complete before any pipeline code.

- [ ] T001 Create project directory structure per plan.md (`src/`, `data/`, `tests/`, `specs/`)
- [ ] T002 Create `requirements.txt` with all dependencies (langchain, chromadb, sentence-transformers, openai, pydantic, fastapi, uvicorn, pytest, python-dotenv)
- [ ] T003 [P] Create `.env.example` with required env vars (`OPENAI_API_BASE`, `OPENAI_API_KEY`, `EMBEDDING_MODEL`, `CHROMA_PERSIST_DIR`)
- [ ] T004 [P] Create `README.md` with project overview, setup steps, and quickstart
- [ ] T005 Create `data/rubrics/grade_6/argumentative.md` — Grade 6 argumentative rubric (6 criteria × 4 levels) with CCSS metadata header
- [ ] T006 [P] Create `data/rubrics/grade_6/informative.md` — Grade 6 informative rubric with CCSS metadata header
- [ ] T007 [P] Create `data/rubrics/grade_6/narrative.md` — Grade 6 narrative rubric with CCSS metadata header
- [ ] T008 Create `tests/golden/` directory with at least 3 essays per genre (12 essays total: P1–P4 per genre) and corresponding human scores in JSON
- [ ] T009 [P] Create `pyproject.toml` or `setup.cfg` for the `src` package

**Checkpoint**: All rubric source files and golden essay set exist; dependencies installable with `pip install -r requirements.txt`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data models and vector store infrastructure that ALL user stories depend on.

**⚠️ CRITICAL**: No user story work begins until this phase is complete.

- [ ] T010 Implement `src/models/essay.py` — `Essay` Pydantic model (`grade: int`, `genre: Literal["argumentative","informative","narrative"]`, `text: str`)
- [ ] T011 [P] Implement `src/models/rubric.py` — `RubricCriterion` model (name, standard, grade, genre, level_descriptors: dict[int, str])
- [ ] T012 [P] Implement `src/models/result.py` — `CriterionScore` (criterion_name, level, descriptor, rationale), `AssessmentScore` (genre, standard, criteria: list[CriterionScore], total, performance_level), `StudentReport` (grade, assessments, aggregate_total, overall_level)
- [ ] T013 Implement `src/scoring/validator.py` — validate grade (6–8), genre enum, non-empty essay; output score clamping (1–4); return structured `ValidationError`
- [ ] T014 Implement `src/retrieval/store.py` — Chroma client initialization, collection-per-grade setup (`rubrics_grade_6/7/8`), collection creation/retrieval helper
- [ ] T015 Implement `src/ingestion/loader.py` — load rubric Markdown files from `data/rubrics/<grade>/` and parse into structured chunks with metadata (`grade`, `genre`, `standard`, `criterion`, `source`)
- [ ] T016 Implement `src/ingestion/chunker.py` — split rubric content per criterion, attach metadata, return `list[Document]`
- [ ] T017 Implement `src/ingestion/ingest.py` — CLI entry point: embed chunks using configured embedding model, upsert to Chroma collection; idempotent (skip if already ingested)
- [ ] T018 Implement `src/retrieval/retriever.py` — `get_retriever(grade, genre)` returning a LangChain retriever with Chroma metadata filters (`grade` + `genre`)

**Checkpoint**: Running `python -m src.ingestion.ingest` populates the Grade 6 Chroma collections; `get_retriever(6, "argumentative")` returns rubric chunks.

---

## Phase 3: User Story 1 — Score a Grade 6 Essay Against a Rubric (Priority: P1) 🎯 MVP

**Goal**: `score_essay(grade=6, genre, essay_text)` returns a full `AssessmentScore` with six criterion scores, each citing retrieved rubric text.

**Independent Test**: Run golden set for Grade 6 (all three genres); verify ≥85% criteria agree with human scores within ±1 point.

### Implementation for User Story 1

- [ ] T019 Implement `src/scoring/prompt.py` — LangChain prompt template that injects: retrieved rubric descriptors (all 6 criteria), student essay, and instruction to score each criterion 1–4 with rationale
- [ ] T020 Implement `src/scoring/scorer.py` — `score_essay(grade, genre, essay_text) → AssessmentScore`: (1) validate inputs, (2) retrieve rubric chunks, (3) build prompt, (4) invoke LLM, (5) parse + validate output, (6) clamp scores, (7) return `AssessmentScore`
- [ ] T021 Implement LLM output parser in `src/scoring/scorer.py` — parse structured JSON response from LLM into `list[CriterionScore]`; handle malformed JSON with retry (max 2 retries)
- [ ] T022 Implement `src/cli.py` — CLI wrapper: `python -m src.cli --grade 6 --genre argumentative --file essay.txt` prints scoring result as JSON
- [ ] T023 Write unit tests in `tests/unit/test_scoring.py` — mock retriever and LLM; test prompt construction, output parsing, score clamping, validation error paths
- [ ] T024 Write integration test in `tests/integration/test_score_essay.py` — run `score_essay` against Grade 6 golden essay set; assert agreement threshold (≥85% within ±1)

**Checkpoint**: `python -m src.cli --grade 6 --genre argumentative --file essay.txt` returns a valid JSON `AssessmentScore`; golden set integration test passes.

---

## Phase 4: User Story 2 — Total Score and Performance Level (Priority: P2)

**Goal**: `AssessmentScore` includes correct total (sum of 6 criteria, max 24) and performance level label.

**Independent Test**: Given known criterion scores, total and performance level match the rubric scale table exactly.

### Implementation for User Story 2

- [ ] T025 Add `compute_total_and_level(criteria: list[CriterionScore], max_score=24) → tuple[int, str]` to `src/models/result.py` using the performance scale (90–100% = Advanced, 75–89% = Proficient, 50–74% = Developing, 25–49% = Beginning)
- [ ] T026 Wire `compute_total_and_level` into `score_essay` in `src/scoring/scorer.py` — populate `AssessmentScore.total` and `AssessmentScore.performance_level`
- [ ] T027 Write unit tests in `tests/unit/test_models.py` — parametrize all four performance level bands; assert boundary values (scores 18, 36, 54, 65, 72)

**Checkpoint**: `AssessmentScore` output includes `total` and `performance_level`; boundary unit tests all pass.

---

## Phase 5: User Story 3 — Aggregate Scoring Across All Three Assessments (Priority: P3)

**Goal**: `score_student(grade, essays: dict[genre, text]) → StudentReport` scoring all three genres and returning aggregate total out of 72 and overall level.

**Independent Test**: Three essays scored and summed; overall level matches 72-point scale.

### Implementation for User Story 3

- [ ] T028 Implement `score_student(grade: int, essays: dict[str, str]) → StudentReport` in `src/scoring/scorer.py` — calls `score_essay` for each genre, assembles `StudentReport` with three `AssessmentScore` objects
- [ ] T029 Add `compute_aggregate(assessments: list[AssessmentScore]) → tuple[int, str]` to `src/models/result.py` — sum totals (max 72), map to overall performance level using the 72-point scale
- [ ] T030 Extend `src/cli.py` — add `--all` flag: accepts three `--file` arguments (one per genre), calls `score_student`, prints full `StudentReport` as JSON
- [ ] T031 Write unit tests in `tests/unit/test_models.py` — test `compute_aggregate` across all four performance bands of the 72-point scale
- [ ] T032 Write integration test in `tests/integration/test_score_essay.py` — `score_student` with three Grade 6 golden essays returns correct `StudentReport` aggregate

**Checkpoint**: `python -m src.cli --all --grade 6 --argumentative arg.txt --informative inf.txt --narrative nar.txt` returns a valid `StudentReport`.

---

## Phase 6: User Story 4 — Multi-Grade Support (Grades 7 and 8) (Priority: P4)

**Goal**: Same pipeline works for Grades 7 and 8 by retrieving from the correct grade-scoped Chroma collection.

**Independent Test**: `score_essay(7, "argumentative", essay)` retrieves Grade 7 rubric chunks (not Grade 6).

### Implementation for User Story 4

- [ ] T033 Create `data/rubrics/grade_7/` and `data/rubrics/grade_8/` rubric Markdown files (argumentative, informative, narrative for each — populate from Grade 7/8 CCSS rubric sources)
- [ ] T034 [P] Update `src/ingestion/ingest.py` to loop over all grade directories and populate `rubrics_grade_7` and `rubrics_grade_8` collections
- [ ] T035 [P] Update `src/retrieval/retriever.py` — ensure `get_retriever(grade, genre)` selects the correct Chroma collection for grades 7 and 8
- [ ] T036 Write integration tests in `tests/integration/test_score_essay.py` — `score_essay(7, "argumentative", essay)` and `score_essay(8, "argumentative", essay)` return rubric chunks with matching grade metadata

**Checkpoint**: All three grade bands ingest and score independently; metadata on retrieved chunks shows correct grade.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, logging, error observability, and quickstart validation.

- [ ] T037 [P] Add structured logging (`logging` stdlib) to `src/scoring/scorer.py` — log grade, genre, retrieval hit count, LLM call duration, retry count
- [ ] T038 [P] Add `src/api/main.py` — FastAPI endpoint `POST /score` accepting `{grade, genre, essay_text}` returning `AssessmentScore`; `POST /score/all` returning `StudentReport`
- [ ] T039 [P] Write `specs/001-rag-writing-assessment/quickstart.md` — step-by-step: install deps, set env vars, run ingestion, score an essay via CLI and API
- [ ] T040 [P] Write `specs/001-rag-writing-assessment/data-model.md` — full Pydantic model reference with field descriptions and example JSON
- [ ] T041 Run full golden set integration test suite; document agreement scores in `specs/001-rag-writing-assessment/research.md`
- [ ] T042 [P] Security hardening — ensure API key / base URL never logged; validate essay text length (max 5000 chars)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — BLOCKS all user stories
- **Phase 3 (US1)**: Depends on Phase 2 — MVP blocker
- **Phase 4 (US2)**: Depends on Phase 3 (needs `CriterionScore` list from scorer)
- **Phase 5 (US3)**: Depends on Phase 4 (needs working `AssessmentScore` with total + level)
- **Phase 6 (US4)**: Depends on Phase 2; can proceed in parallel with Phase 3+ once ingestion is working
- **Phase 7 (Polish)**: Depends on all user story phases complete

### Parallel Opportunities

- T005, T006, T007 — rubric files can be authored in parallel
- T010, T011, T012 — Pydantic models have no internal dependencies
- T014, T015, T016 — store, loader, chunker are independent
- T033, T034, T035 — Grade 7/8 ingestion work is parallelizable
- T037, T038, T039, T040, T042 — polish tasks are all independent

---

## Implementation Strategy

### MVP (Phase 1 → Phase 3 only)

1. Phase 1: Setup + rubric files + golden set
2. Phase 2: Models + ingestion + retriever
3. Phase 3: Scorer + CLI + golden set test
4. **STOP and VALIDATE**: Golden set integration test must pass before proceeding

### Incremental Delivery

- MVP: Grade 6, single genre score via CLI ← **start here**
- v0.2: Add total + performance level
- v0.3: Full three-genre student report
- v0.4: Grade 7 and 8 support
- v0.5: FastAPI endpoint + logging
