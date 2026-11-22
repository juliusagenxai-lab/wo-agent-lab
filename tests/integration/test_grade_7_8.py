"""
Integration tests for Grade 7 and Grade 8 retrieval — Phase 6 (T036/T037).

Verifies that:
- retrieve_rubric_chunks returns chunks with correct grade metadata for grades 7 and 8
- The correct collection is selected (no cross-grade contamination)
- All 6 criteria are returned per genre for each grade
- score_essay() correctly routes to grade-scoped rubrics for grades 7 and 8
"""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage


# ── Helpers ───────────────────────────────────────────────────────────────────

GRADE_7_CRITERIA = {
    "argumentative": ["Claim", "Evidence", "Reasoning", "Organization", "Counterargument", "Language Conventions"],
    "informative":   ["Topic Introduction", "Development of Ideas", "Facts and Examples", "Organization", "Conclusion", "Language Conventions"],
    "narrative":     ["Plot Development", "Characters", "Setting", "Organization", "Descriptive Language", "Language Conventions"],
}
GRADE_8_CRITERIA = GRADE_7_CRITERIA  # same criterion names, different descriptors

GRADE_STANDARDS = {
    7: {"argumentative": "W.7.1", "informative": "W.7.2", "narrative": "W.7.3"},
    8: {"argumentative": "W.8.1", "informative": "W.8.2", "narrative": "W.8.3"},
}

SAMPLE_ESSAY_G7 = (
    "Social media has become a central part of teenagers' lives, but its effects on mental health "
    "are concerning. Studies show that excessive social media use is linked to increased anxiety and "
    "depression in adolescents. Schools and parents should help young people develop healthier digital "
    "habits before the problem worsens. While social media can connect people, the documented harms "
    "to adolescent mental health outweigh the benefits when use is unregulated."
)

SAMPLE_ESSAY_G8 = (
    "The widespread adoption of artificial intelligence in hiring processes raises serious ethical "
    "questions about fairness and accountability. Algorithmic bias has been documented in multiple "
    "high-profile cases, disproportionately disadvantaging applicants from underrepresented groups. "
    "Organizations must implement transparent oversight mechanisms and regular audits to ensure AI "
    "hiring tools uphold principles of equity. Failing to act risks entrenching systemic discrimination "
    "at scale under the guise of objectivity."
)


def _make_valid_llm_response(criteria: list[str], level: int = 3) -> str:
    return json.dumps([
        {
            "criterion_name": c,
            "level": level,
            "descriptor": f"Level {level} descriptor for {c}.",
            "rationale": f"The essay demonstrates level {level} for {c}.",
        }
        for c in criteria
    ])


def _make_mock_llm(response: str) -> MagicMock:
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content=response)
    return mock


def _make_mock_docs(grade: int, genre: str, criteria: list[str]) -> list[Document]:
    standard = GRADE_STANDARDS[grade][genre]
    return [
        Document(
            page_content=(
                f"Grade {grade} {genre.capitalize()} Writing – Criterion: {c}\n"
                f"  Level 4 (Advanced): Advanced {c.lower()}.\n"
                f"  Level 3 (Proficient): Proficient {c.lower()}.\n"
                f"  Level 2 (Developing): Developing {c.lower()}.\n"
                f"  Level 1 (Beginning): Beginning {c.lower()}.\n"
            ),
            metadata={"criterion": c, "genre": genre, "grade": grade, "standard": standard},
        )
        for c in criteria
    ]


# ── T036: Retriever grade isolation tests (live vector store) ─────────────────

class TestGrade7And8Retrieval:
    """
    Verifies the retriever correctly isolates grade 7 and grade 8 collections.
    Requires the vector store to be ingested (python -m src.ingestion.ingest --grades 7 8).
    """

    @pytest.mark.parametrize("grade,genre,expected_criteria", [
        (7, "argumentative", GRADE_7_CRITERIA["argumentative"]),
        (7, "informative",   GRADE_7_CRITERIA["informative"]),
        (7, "narrative",     GRADE_7_CRITERIA["narrative"]),
        (8, "argumentative", GRADE_8_CRITERIA["argumentative"]),
        (8, "informative",   GRADE_8_CRITERIA["informative"]),
        (8, "narrative",     GRADE_8_CRITERIA["narrative"]),
    ])
    def test_retriever_returns_correct_grade_metadata(self, grade, genre, expected_criteria):
        """All retrieved chunks must carry the correct grade in their metadata."""
        from src.retrieval.retriever import retrieve_rubric_chunks

        docs = retrieve_rubric_chunks(grade=grade, genre=genre)

        assert len(docs) == 6, f"Expected 6 chunks for grade={grade} genre={genre}, got {len(docs)}"

        for doc in docs:
            assert doc.metadata.get("grade") == grade, (
                f"Wrong grade in metadata: expected {grade}, got {doc.metadata.get('grade')}"
            )
            assert doc.metadata.get("genre") == genre, (
                f"Wrong genre in metadata: expected {genre}, got {doc.metadata.get('genre')}"
            )

    @pytest.mark.parametrize("grade,genre", [
        (7, "argumentative"), (7, "informative"), (7, "narrative"),
        (8, "argumentative"), (8, "informative"), (8, "narrative"),
    ])
    def test_retriever_returns_six_unique_criteria(self, grade, genre):
        """Retriever returns exactly 6 unique criterion chunks for each grade+genre."""
        from src.retrieval.retriever import retrieve_rubric_chunks

        docs = retrieve_rubric_chunks(grade=grade, genre=genre)
        criteria_names = [d.metadata.get("criterion") for d in docs]

        assert len(set(criteria_names)) == 6, (
            f"Expected 6 unique criteria for grade={grade} genre={genre}, got: {criteria_names}"
        )

    @pytest.mark.parametrize("grade,genre,standard", [
        (7, "argumentative", "W.7.1"),
        (7, "informative",   "W.7.2"),
        (7, "narrative",     "W.7.3"),
        (8, "argumentative", "W.8.1"),
        (8, "informative",   "W.8.2"),
        (8, "narrative",     "W.8.3"),
    ])
    def test_retriever_returns_correct_standard_metadata(self, grade, genre, standard):
        """All retrieved chunks must carry the correct CCSS standard."""
        from src.retrieval.retriever import retrieve_rubric_chunks

        docs = retrieve_rubric_chunks(grade=grade, genre=genre)

        for doc in docs:
            assert doc.metadata.get("standard") == standard, (
                f"Wrong standard for grade={grade} genre={genre}: "
                f"expected {standard}, got {doc.metadata.get('standard')}"
            )

    def test_grade_7_and_8_stores_are_independent(self):
        """Grade 7 and Grade 8 vector stores must not share documents."""
        from src.retrieval.retriever import retrieve_rubric_chunks

        docs_7 = retrieve_rubric_chunks(grade=7, genre="argumentative")
        docs_8 = retrieve_rubric_chunks(grade=8, genre="argumentative")

        grades_7 = {d.metadata.get("grade") for d in docs_7}
        grades_8 = {d.metadata.get("grade") for d in docs_8}

        assert grades_7 == {7}, f"Grade 7 store leaked: grades found = {grades_7}"
        assert grades_8 == {8}, f"Grade 8 store leaked: grades found = {grades_8}"

    def test_grade_7_descriptors_differ_from_grade_6(self):
        """Grade 7 descriptor text is not identical to Grade 6 (rubrics are grade-specific)."""
        from src.retrieval.retriever import retrieve_rubric_chunks

        docs_6 = retrieve_rubric_chunks(grade=6, genre="argumentative")
        docs_7 = retrieve_rubric_chunks(grade=7, genre="argumentative")

        content_6 = {d.metadata["criterion"]: d.page_content for d in docs_6}
        content_7 = {d.metadata["criterion"]: d.page_content for d in docs_7}

        for criterion in content_6:
            if criterion in content_7:
                assert content_6[criterion] != content_7[criterion], (
                    f"Criterion '{criterion}' has identical content in grade 6 and grade 7 — "
                    "rubric files may be duplicated"
                )


# ── T037: score_essay() grade routing tests (mocked retriever) ────────────────

class TestScoreEssayGrade7And8:
    """
    Verifies score_essay() correctly routes to grade-scoped rubrics for grades 7 and 8.
    Uses mocked retriever and LLM — no live endpoint required.
    """

    @pytest.mark.parametrize("grade,genre,essay", [
        (7, "argumentative", SAMPLE_ESSAY_G7),
        (7, "informative",   SAMPLE_ESSAY_G7),
        (7, "narrative",     SAMPLE_ESSAY_G7),
        (8, "argumentative", SAMPLE_ESSAY_G8),
        (8, "informative",   SAMPLE_ESSAY_G8),
        (8, "narrative",     SAMPLE_ESSAY_G8),
    ])
    def test_score_essay_returns_assessment_score_for_grade_7_8(self, grade, genre, essay):
        """score_essay returns a valid AssessmentScore for grades 7 and 8."""
        from src.scoring.scorer import score_essay
        from src.models.result import AssessmentScore

        criteria = GRADE_7_CRITERIA[genre]  # same names for both grades
        mock_docs = _make_mock_docs(grade, genre, criteria)
        mock_llm = _make_mock_llm(_make_valid_llm_response(criteria))

        with patch("src.scoring.scorer.retrieve_rubric_chunks") as mock_retrieve:
            mock_retrieve.return_value = mock_docs
            result = score_essay(grade=grade, genre=genre, essay_text=essay, llm=mock_llm)

        assert isinstance(result, AssessmentScore)
        assert result.grade == grade
        assert result.genre == genre
        assert result.standard == GRADE_STANDARDS[grade][genre]
        assert len(result.criteria) == 6
        assert 6 <= result.total <= 24

    @pytest.mark.parametrize("grade,genre", [
        (7, "argumentative"), (7, "informative"), (7, "narrative"),
        (8, "argumentative"), (8, "informative"), (8, "narrative"),
    ])
    def test_retrieve_called_with_correct_grade(self, grade, genre):
        """Constitution check: retriever is called with the exact grade parameter."""
        from src.scoring.scorer import score_essay

        criteria = GRADE_7_CRITERIA[genre]
        mock_docs = _make_mock_docs(grade, genre, criteria)
        mock_llm = _make_mock_llm(_make_valid_llm_response(criteria))

        with patch("src.scoring.scorer.retrieve_rubric_chunks") as mock_retrieve:
            mock_retrieve.return_value = mock_docs
            score_essay(grade=grade, genre=genre, essay_text=SAMPLE_ESSAY_G7, llm=mock_llm)

        mock_retrieve.assert_called_once_with(grade=grade, genre=genre)

    @pytest.mark.parametrize("grade,genre", [
        (7, "argumentative"), (8, "argumentative"),
    ])
    def test_standard_in_output_matches_grade(self, grade, genre):
        """AssessmentScore.standard must match the grade-specific CCSS standard."""
        from src.scoring.scorer import score_essay

        criteria = GRADE_7_CRITERIA[genre]
        mock_docs = _make_mock_docs(grade, genre, criteria)
        mock_llm = _make_mock_llm(_make_valid_llm_response(criteria))

        with patch("src.scoring.scorer.retrieve_rubric_chunks") as mock_retrieve:
            mock_retrieve.return_value = mock_docs
            result = score_essay(grade=grade, genre=genre, essay_text=SAMPLE_ESSAY_G8, llm=mock_llm)

        expected = GRADE_STANDARDS[grade][genre]
        assert result.standard == expected, (
            f"Expected standard {expected} for grade {grade}, got {result.standard}"
        )

    def test_grade_7_and_8_produce_independent_assessments(self):
        """Scoring the same essay at grade 7 vs grade 8 retrieves from different stores."""
        from src.scoring.scorer import score_essay

        criteria = GRADE_7_CRITERIA["argumentative"]
        docs_7 = _make_mock_docs(7, "argumentative", criteria)
        docs_8 = _make_mock_docs(8, "argumentative", criteria)

        llm_7 = _make_mock_llm(_make_valid_llm_response(criteria, level=3))
        llm_8 = _make_mock_llm(_make_valid_llm_response(criteria, level=4))

        with patch("src.scoring.scorer.retrieve_rubric_chunks") as mock_retrieve:
            mock_retrieve.return_value = docs_7
            result_7 = score_essay(grade=7, genre="argumentative",
                                   essay_text=SAMPLE_ESSAY_G7, llm=llm_7)

        with patch("src.scoring.scorer.retrieve_rubric_chunks") as mock_retrieve:
            mock_retrieve.return_value = docs_8
            result_8 = score_essay(grade=8, genre="argumentative",
                                   essay_text=SAMPLE_ESSAY_G8, llm=llm_8)

        assert result_7.grade == 7
        assert result_8.grade == 8
        assert result_7.standard == "W.7.1"
        assert result_8.standard == "W.8.1"
        # Different LLM responses → different totals
        assert result_7.total != result_8.total
