"""
Main scoring pipeline.

Public interface:
    score_essay(grade, genre, essay_text) -> AssessmentScore
    score_student(grade, essays)          -> StudentReport

Constitution requirement: retrieval MUST occur before every LLM call.
The scorer raises ValueError immediately if validation fails, before
any retrieval or LLM call is made.
"""
import json
import logging
import os
import re

from dotenv import load_dotenv

from src.models.result import AssessmentScore, CriterionScore, StudentReport
from src.retrieval.retriever import retrieve_rubric_chunks
from src.scoring.prompt import build_scoring_prompt, get_standard
from src.scoring.validator import validate_inputs

load_dotenv()

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2


# ── LLM factory ──────────────────────────────────────────────────────────────

def _build_llm():
    """
    Build and return the LangChain LLM from environment config.

    Supported providers (set LLM_PROVIDER env var):
      "gemini"  → ChatGoogleGenerativeAI  (default; requires GOOGLE_API_KEY)
      "openai"  → ChatOpenAI / Ollama     (requires OPENAI_API_BASE + OPENAI_API_KEY)
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model = os.getenv("LLM_MODEL", "gemini-3-flash-preview")

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key or api_key == "your-google-api-key-here":
            raise ValueError(
                "GOOGLE_API_KEY is not set. "
                "Get a free key at https://aistudio.google.com/app/apikey "
                "and add it to your .env file."
            )
        logger.info("Building Gemini LLM: model=%s", model)
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0,
            max_output_tokens=4096,
            response_mime_type="application/json",
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        api_base = os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
        api_key = os.getenv("OPENAI_API_KEY", "ollama")
        logger.info("Building OpenAI-compatible LLM: model=%s base=%s", model, api_base)
        return ChatOpenAI(
            model_name=model,
            openai_api_base=api_base,
            openai_api_key=api_key,
            temperature=0,
            max_tokens=2048,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider!r}. Valid options: 'gemini', 'openai'"
        )


# ── Output parser ─────────────────────────────────────────────────────────────

def _parse_llm_output(raw: str, rubric_docs: list) -> list[dict]:
    """
    Parse the LLM's raw response string into a list of criterion score dicts.

    Strips markdown code fences if present, then JSON-parses.
    Returns a list of dicts with keys: criterion_name, level, descriptor, rationale.

    Raises:
        ValueError: If the JSON cannot be parsed or is not a list.
    """
    # Gemini may return content as a list of parts — flatten to string
    if isinstance(raw, list):
        raw = " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in raw
        )

    # Strip markdown fences: ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM output is not valid JSON: {exc}\nRaw output:\n{raw}") from exc

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}. Raw:\n{raw}")

    return data


def _dicts_to_criterion_scores(
    dicts: list[dict],
    rubric_docs: list,
) -> list:
    """Convert parsed LLM dicts to CriterionScore objects, clamping levels."""
    from src.models.result import CriterionScore
    from src.scoring.validator import clamp_level

    scores = []
    for item in dicts:
        raw_level = item.get("level", 1)
        criterion_name = item.get("criterion_name", "Unknown")
        level = clamp_level(raw_level, criterion_name)

        scores.append(CriterionScore(
            criterion_name=criterion_name,
            level=level,
            descriptor=str(item.get("descriptor", "")),
            rationale=str(item.get("rationale", "")),
        ))

    return scores


# ── Core scorer ───────────────────────────────────────────────────────────────

def score_essay(
    grade: int,
    genre: str,
    essay_text: str,
    llm=None,
) -> "AssessmentScore":  # noqa: F821
    """
    Score a single student essay against the matching CCSS rubric.

    Args:
        grade:      Student grade level (6, 7, or 8).
        genre:      Writing genre — "argumentative", "informative", or "narrative".
        essay_text: The student's essay as a plain-text string.
        llm:        Optional pre-built LLM (for testing); built from env vars if None.

    Returns:
        AssessmentScore with six CriterionScores, total, and performance level.

    Raises:
        ValueError:       If input validation fails (bad grade, genre, or empty text).
        FileNotFoundError: If the vector store for the given grade is not ingested.
        RuntimeError:     If the LLM fails to return valid JSON after max retries.
    """
    # 1. Validate inputs — fail fast before any expensive operations
    errors = validate_inputs(grade, genre, essay_text)
    if errors:
        messages = "; ".join(f"[{e.field}] {e.message}" for e in errors)
        raise ValueError(f"Input validation failed: {messages}")

    # 2. Retrieve rubric chunks (constitution: MUST happen before LLM call)
    logger.info("Retrieving rubric chunks: grade=%d genre=%s", grade, genre)
    rubric_docs = retrieve_rubric_chunks(grade=grade, genre=genre)
    if not rubric_docs:
        raise RuntimeError(
            f"No rubric chunks retrieved for grade={grade} genre={genre}. "
            "Re-run ingestion: python -m src.ingestion.ingest"
        )

    standard = get_standard(grade, genre)

    # 3. Build prompt
    messages = build_scoring_prompt(
        grade=grade,
        genre=genre,
        standard=standard,
        rubric_docs=rubric_docs,
        essay_text=essay_text,
    )

    # 4. Invoke LLM with retry on bad JSON
    if llm is None:
        llm = _build_llm()

    raw_output = ""
    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.info("Invoking LLM (attempt %d/%d)", attempt, _MAX_RETRIES)
            response = llm.invoke(messages)
            raw_output = response.content
            criterion_dicts = _parse_llm_output(raw_output, rubric_docs)
            break
        except (ValueError, AttributeError) as exc:
            last_error = exc
            logger.warning("LLM parse attempt %d failed: %s", attempt, exc)
            if attempt == _MAX_RETRIES:
                raise RuntimeError(
                    f"LLM failed to return valid JSON after {_MAX_RETRIES} attempts. "
                    f"Last error: {last_error}\nLast output:\n{raw_output}"
                ) from last_error

    # 5. Convert to CriterionScore objects, clamp levels
    criteria = _dicts_to_criterion_scores(criterion_dicts, rubric_docs)

    # 6. Build AssessmentScore and compute total + level
    assessment = AssessmentScore(
        grade=grade,
        genre=genre,
        standard=standard,
        criteria=criteria,
    )
    assessment.compute_total_and_level()

    logger.info(
        "Scored grade=%d genre=%s total=%d level=%s",
        grade, genre, assessment.total, assessment.performance_level,
    )
    return assessment


def score_student(
    grade: int,
    essays: dict[str, str],
    llm=None,
) -> "StudentReport":  # noqa: F821
    """
    Score all three genre essays for one student and return an aggregate report.

    Args:
        grade:  Student grade level (6, 7, or 8).
        essays: Dict mapping genre → essay_text.
                e.g. {"argumentative": "...", "informative": "...", "narrative": "..."}
        llm:    Optional pre-built LLM (for testing).

    Returns:
        StudentReport with three AssessmentScores and aggregate total/level.
    """
    if llm is None:
        llm = _build_llm()

    assessments = []
    for genre, text in essays.items():
        logger.info("Scoring genre=%s for grade=%d", genre, grade)
        assessment = score_essay(grade=grade, genre=genre, essay_text=text, llm=llm)
        assessments.append(assessment)

    report = StudentReport(grade=grade, assessments=assessments)
    report.compute_aggregate()
    return report
