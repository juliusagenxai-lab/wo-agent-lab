# Feature Specification: RAG Writing Assessment & Scoring Agent

**Feature Branch**: `001-rag-writing-assessment`

**Created**: 2025-11-22

**Status**: Draft

**Input**: User description — Build a Retrieval Augmented Agent (RAG) using GPT-OSS reasoning for state academic writing assessments and scoring rubrics used by states to evaluate Grades 6–8 students' writing, aligned to U.S. Common Core State Standards (CCSS) / California Common Core State Standards (CA CCSS).

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Score a Grade 6 Essay Against a Rubric (Priority: P1)

A teacher pastes a Grade 6 student essay (argumentative, informative/explanatory, or narrative) and selects the grade and genre. The agent retrieves the matching rubric from the knowledge base and returns a structured score for each of the six rubric criteria (1–4), with the retrieved descriptor text and a brief rationale tied to the student's writing.

**Why this priority**: This is the entire core value proposition. Everything else builds on this working.

**Independent Test**: Given a sample Grade 6 argumentative essay, calling `score_essay(grade=6, genre="argumentative", essay_text=<essay>)` returns a `ScoringResult` with six criterion scores, each citing retrieved rubric text and a one-sentence rationale. Can be validated against a human-scored golden set.

**Acceptance Scenarios**:

1. **Given** a Grade 6 argumentative essay, **When** `score_essay(6, "argumentative", essay)` is called, **Then** the response contains scores for all six criteria (Claim, Evidence, Reasoning, Organization, Counterargument, Language Conventions), each with a level (1–4), retrieved rubric descriptor, and rationale.
2. **Given** a Grade 6 informative essay, **When** `score_essay(6, "informative", essay)` is called, **Then** the response contains scores for all six informative criteria with matching retrieved rubric text.
3. **Given** a Grade 6 narrative essay, **When** `score_essay(6, "narrative", essay)` is called, **Then** the response contains scores for all six narrative criteria with matching retrieved rubric text.
4. **Given** an empty essay string, **When** `score_essay` is called, **Then** the system returns a validation error, not a score.

---

### User Story 2 — Retrieve Total Score and Performance Level (Priority: P2)

After criterion scoring, the agent computes the total score (sum of six criteria, max 24 per assessment), maps it to the overall proficiency level (Advanced / Proficient / Developing / Beginning), and returns the full scoring result with the performance level label.

**Why this priority**: Teachers need a single performance-level summary, not just raw criterion numbers.

**Independent Test**: Given a scoring result with known criterion scores (e.g., 4+3+3+4+3+3 = 20), the total and performance level returned match the rubric scale (20/24 = Proficient).

**Acceptance Scenarios**:

1. **Given** criterion scores summing to 20 for a single assessment, **When** the total is computed, **Then** the performance level is "Proficient" (75–89%).
2. **Given** criterion scores summing to 22, **When** the total is computed, **Then** the performance level is "Advanced" (90–100%).
3. **Given** criterion scores summing to 10, **When** the total is computed, **Then** the performance level is "Beginning" (25–49%).

---

### User Story 3 — Score Across All Three Assessments and Return Aggregate (Priority: P3)

A teacher submits all three essays (argumentative, informative, narrative) for one student. The agent scores each independently and returns an aggregate report: individual scores per assessment, total score out of 72, and an overall performance level.

**Why this priority**: The full assessment framework covers three genres; teachers need a single student report.

**Independent Test**: Given three essays (one per genre), the aggregate report contains three individual scoring results, a total out of 72, and an overall performance level aligned to the scoring table (65–72 = Advanced, 54–64 = Proficient, 36–53 = Developing, 18–35 = Beginning).

**Acceptance Scenarios**:

1. **Given** three essays scoring 20 + 18 + 19 = 57 total, **When** the aggregate is computed, **Then** the overall level is "Proficient" (75–89%).
2. **Given** three essays, **When** the aggregate report is generated, **Then** each individual assessment section references the correct genre rubric and standard (W.6.1, W.6.2, W.6.3).

---

### User Story 4 — Multi-Grade Support (Grades 7 and 8) (Priority: P4)

The same scoring pipeline works for Grade 7 and Grade 8 by retrieving the appropriate grade-specific rubric from the knowledge base. The `grade` parameter controls which rubric collection is queried.

**Why this priority**: MVP is Grade 6; Grades 7 and 8 extend the same architecture without new code paths.

**Independent Test**: Given a Grade 7 argumentative essay, `score_essay(7, "argumentative", essay)` retrieves Grade 7 rubric descriptors (not Grade 6) and returns a valid scoring result.

**Acceptance Scenarios**:

1. **Given** a Grade 7 essay, **When** scored, **Then** retrieved rubric text carries `grade: 7` metadata.
2. **Given** a Grade 8 essay, **When** scored, **Then** retrieved rubric text carries `grade: 8` metadata.

---

### Edge Cases

- What happens when the essay is in a language other than English? → Return a validation error: "Essay must be in English."
- What happens when the grade is out of range (e.g., grade=5)? → Return a validation error: "Supported grades: 6, 7, 8."
- What happens when the genre string is unrecognized? → Return a validation error listing the three valid genres.
- What if the vector store returns no results for a rubric query? → Return a retrieval error; do not attempt to score from model memory.
- What if the LLM returns a score outside the 1–4 range? → Clamp and log a warning; do not surface an out-of-range score to the user.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST ingest and chunk rubric documents for Grades 6, 7, and 8, all three genres, into a vector store with metadata fields: `grade`, `genre`, `standard`, `criterion`, `source`.
- **FR-002**: System MUST retrieve relevant rubric criterion descriptors using a semantic query filtered by `grade` and `genre` before invoking the LLM.
- **FR-003**: System MUST return a score (1–4) for each of the six rubric criteria per assessment.
- **FR-004**: Each criterion score MUST include: criterion name, score level, retrieved rubric descriptor text, and a rationale sentence referencing the student essay.
- **FR-005**: System MUST compute total score per assessment (max 24) and map to performance level.
- **FR-006**: System MUST support aggregate scoring across all three genre assessments, returning a total out of 72 and an overall performance level.
- **FR-007**: System MUST support grades 6, 7, and 8 via parameterized rubric retrieval.
- **FR-008**: System MUST validate inputs (grade range, genre enum, non-empty essay) and return structured errors before invoking retrieval or LLM.
- **FR-009**: System MUST surface the CCSS/CA CCSS standard identifier (e.g., W.6.1) in the scoring output for each assessment.
- **FR-010**: System MUST NOT score from model memory alone — retrieval MUST occur before every scoring call.

### Key Entities

- **Essay**: Student-submitted text with metadata (grade, genre).
- **RubricCriterion**: A single scoring dimension (e.g., "Claim") with descriptors for levels 1–4, tagged with grade, genre, standard.
- **CriterionScore**: The agent's score for one RubricCriterion — level (1–4), retrieved descriptor, rationale.
- **AssessmentScore**: Six CriterionScores + total (max 24) + performance level for one genre assessment.
- **StudentReport**: Three AssessmentScores + aggregate total (max 72) + overall performance level.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Agent scores on the golden essay set agree with human scores within ±1 point per criterion for ≥85% of criteria.
- **SC-002**: Every scoring response includes retrieved rubric text — zero responses scored from model memory.
- **SC-003**: Performance level mapping is 100% correct against the defined scoring table.
- **SC-004**: Input validation rejects all out-of-range grades and unrecognized genres with a structured error.
- **SC-005**: End-to-end `score_essay` latency is under 15 seconds for a 500-word essay on target hardware.

---

## Assumptions

- Grade 6 rubric content (as provided in the prompt) is the MVP; Grades 7 and 8 rubric documents will be provided before Phase 4 (US4) is implemented.
- The LLM is accessed via an OpenAI-compatible API (local OSS model via Ollama or remote endpoint); API credentials are managed via environment variables.
- Rubric content is stored as Markdown documents and ingested during a one-time setup step; real-time rubric editing is out of scope for v1.
- The scoring interface is a Python library/CLI first; a web API (FastAPI) is a Phase 3+ concern.
- A golden essay set will be curated manually before the scoring pipeline is coded (required by the constitution).
- Multi-language support is out of scope for v1.
- Official CAASPP Smarter Balanced assessment content is NOT used; this system uses CCSS-aligned classroom rubrics.
