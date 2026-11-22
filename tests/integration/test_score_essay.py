"""
Integration tests for score_essay() against the golden essay set.

Two modes:
  1. MOCK mode (default, CI-safe): uses a mock LLM that returns pre-defined
     scores matching the golden set. Validates pipeline structure and agreement logic.
  2. LIVE mode (opt-in): set env var RAG_TEST_LIVE=1 to use the real LLM endpoint.
     Requires OPENAI_API_BASE and LLM_MODEL to be configured in .env.
     Validates ≥85% criterion agreement with human scores within ±1 point.

Run mock mode:
    python -m pytest tests/integration/test_score_essay.py -v

Run live mode (requires running LLM):
    RAG_TEST_LIVE=1 python -m pytest tests/integration/test_score_essay.py -v -s
"""
import json
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage


# ── Golden set loader ─────────────────────────────────────────────────────────

GOLDEN_DIR = Path("tests/golden")
GENRES = ["argumentative", "informative", "narrative"]
LEVELS = ["advanced", "proficient", "developing", "beginning"]


def load_golden_case(genre: str, level: str) -> tuple[str, dict]:
    """Return (essay_text, human_scores_dict) for a given genre and level."""
    essay_path = GOLDEN_DIR / f"grade_6_{genre}" / f"essay_{level}.txt"
    scores_path = GOLDEN_DIR / f"grade_6_{genre}" / f"scores_{level}.json"
    essay_text = essay_path.read_text(encoding="utf-8")
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    return essay_text, scores


def human_criterion_levels(scores: dict) -> dict[str, int]:
    """Return {criterion_name: level} from a human scores dict."""
    return {c["criterion"]: c["level"] for c in scores["criteria"]}


# ── Mock LLM that returns human-score-matching responses ─────────────────────

def _build_mock_response(human_scores: dict) -> str:
    """Build a JSON LLM response that matches the human scores exactly."""
    criteria = human_scores["criteria"]
    response = [
        {
            "criterion_name": c["criterion"],
            "level": c["level"],
            "descriptor": f"Level {c['level']} descriptor for {c['criterion']}.",
            "rationale": f"The essay demonstrates level {c['level']} for {c['criterion']}.",
        }
        for c in criteria
    ]
    return json.dumps(response)


def _make_mock_llm_for_case(human_scores: dict):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=_build_mock_response(human_scores))
    return mock_llm


# ── Agreement metric ──────────────────────────────────────────────────────────

def compute_agreement(agent_criteria, human_levels: dict[str, int]) -> dict:
    """
    Compare agent criterion scores against human scores.
    Returns agreement stats: total criteria, within-1 count, agreement rate.
    """
    total = 0
    within_1 = 0
    details = []
    for cs in agent_criteria:
        human_level = human_levels.get(cs.criterion_name)
        if human_level is None:
            continue
        diff = abs(cs.level - human_level)
        within_1 += int(diff <= 1)
        total += 1
        details.append({
            "criterion": cs.criterion_name,
            "agent": cs.level,
            "human": human_level,
            "diff": diff,
            "within_1": diff <= 1,
        })
    return {
        "total": total,
        "within_1": within_1,
        "rate": within_1 / total if total > 0 else 0.0,
        "details": details,
    }


# ── Mock-mode integration tests ───────────────────────────────────────────────

class TestGoldenSetMock:
    """
    Validates pipeline structure using mock LLM responses derived from human scores.
    These tests do NOT require a live LLM.
    """

    @pytest.mark.parametrize("genre,level", [
        (genre, level) for genre in GENRES for level in LEVELS
    ])
    def test_score_essay_returns_assessment_score(self, genre, level):
        """score_essay returns a valid AssessmentScore for every golden essay."""
        from src.scoring.scorer import score_essay
        from src.models.result import AssessmentScore

        essay_text, human_scores = load_golden_case(genre, level)
        mock_llm = _make_mock_llm_for_case(human_scores)

        with patch("src.scoring.scorer.retrieve_rubric_chunks") as mock_retrieve:
            # Provide minimal rubric docs so prompt builds without hitting disk
            from langchain_core.documents import Document
            mock_docs = [
                Document(
                    page_content=(
                        f"Grade 6 {genre.capitalize()} Writing – Criterion: {c['criterion']}\n"
                        f"  Level 4 (Advanced): Advanced.\n"
                        f"  Level 3 (Proficient): Proficient.\n"
                        f"  Level 2 (Developing): Developing.\n"
                        f"  Level 1 (Beginning): Beginning.\n"
                    ),
                    metadata={"criterion": c["criterion"], "genre": genre,
                               "grade": 6, "standard": human_scores["standard"]},
                )
                for c in human_scores["criteria"]
            ]
            mock_retrieve.return_value = mock_docs

            result = score_essay(grade=6, genre=genre, essay_text=essay_text, llm=mock_llm)

        assert isinstance(result, AssessmentScore)
        assert result.grade == 6
        assert result.genre == genre
        assert result.standard == human_scores["standard"]
        assert len(result.criteria) == 6
        assert 6 <= result.total <= 24
        assert result.performance_level in ("Advanced", "Proficient", "Developing", "Beginning")

    @pytest.mark.parametrize("genre,level", [
        (genre, level) for genre in GENRES for level in LEVELS
    ])
    def test_pipeline_agreement_with_human_scores(self, genre, level):
        """When LLM returns human scores, agreement is 100% (pipeline correctness check)."""
        from src.scoring.scorer import score_essay

        essay_text, human_scores = load_golden_case(genre, level)
        mock_llm = _make_mock_llm_for_case(human_scores)
        human_levels = human_criterion_levels(human_scores)

        with patch("src.scoring.scorer.retrieve_rubric_chunks") as mock_retrieve:
            from langchain_core.documents import Document
            mock_docs = [
                Document(
                    page_content=f"Criterion: {c['criterion']}\n  Level {c['level']}: desc.\n",
                    metadata={"criterion": c["criterion"], "genre": genre, "grade": 6,
                               "standard": human_scores["standard"]},
                )
                for c in human_scores["criteria"]
            ]
            mock_retrieve.return_value = mock_docs
            result = score_essay(grade=6, genre=genre, essay_text=essay_text, llm=mock_llm)

        agreement = compute_agreement(result.criteria, human_levels)
        # With mock LLM returning exact human scores, agreement must be 100%
        assert agreement["rate"] == 1.0, (
            f"Pipeline agreement failed for {genre}/{level}: {agreement['details']}"
        )

    def test_total_matches_human_total_for_advanced_argumentative(self):
        """Spot-check: pipeline total matches human total for advanced argumentative."""
        from src.scoring.scorer import score_essay

        essay_text, human_scores = load_golden_case("argumentative", "advanced")
        mock_llm = _make_mock_llm_for_case(human_scores)

        with patch("src.scoring.scorer.retrieve_rubric_chunks") as mock_retrieve:
            from langchain_core.documents import Document
            mock_docs = [
                Document(
                    page_content=f"Criterion: {c['criterion']}\n",
                    metadata={"criterion": c["criterion"], "genre": "argumentative",
                               "grade": 6, "standard": "W.6.1"},
                )
                for c in human_scores["criteria"]
            ]
            mock_retrieve.return_value = mock_docs
            result = score_essay(grade=6, genre="argumentative", essay_text=essay_text, llm=mock_llm)

        assert result.total == human_scores["total"]
        assert result.performance_level == human_scores["performance_level"]

    def test_retrieval_called_once_per_score_essay_call(self):
        """Constitution check: retrieval is called exactly once per score_essay call."""
        from src.scoring.scorer import score_essay

        essay_text, human_scores = load_golden_case("narrative", "proficient")
        mock_llm = _make_mock_llm_for_case(human_scores)

        with patch("src.scoring.scorer.retrieve_rubric_chunks") as mock_retrieve:
            from langchain_core.documents import Document
            mock_retrieve.return_value = [
                Document(
                    page_content=f"Criterion: {c['criterion']}\n",
                    metadata={"criterion": c["criterion"], "genre": "narrative",
                               "grade": 6, "standard": "W.6.3"},
                )
                for c in human_scores["criteria"]
            ]
            score_essay(grade=6, genre="narrative", essay_text=essay_text, llm=mock_llm)

        mock_retrieve.assert_called_once_with(grade=6, genre="narrative")


# ── Live-mode integration tests (opt-in) ─────────────────────────────────────

LIVE_MODE = os.getenv("RAG_TEST_LIVE", "0") == "1"


@pytest.mark.skipif(not LIVE_MODE, reason="Set RAG_TEST_LIVE=1 to run live LLM tests")
class TestGoldenSetLive:
    """
    Runs score_essay against a live LLM endpoint and validates agreement with
    human scores (≥85% of criteria within ±1 point, per constitution SC-001).

    Requires:
        - Ingested vector store (python -m src.ingestion.ingest --grades 6)
        - .env configured with OPENAI_API_BASE, OPENAI_API_KEY, LLM_MODEL
    """

    @pytest.mark.parametrize("genre,level", [
        (genre, level) for genre in GENRES for level in LEVELS
    ])
    def test_live_agreement_within_1_point(self, genre, level):
        """Live LLM scores must agree with human scores within ±1 for ≥85% of criteria."""
        from src.scoring.scorer import score_essay

        essay_text, human_scores = load_golden_case(genre, level)
        human_levels = human_criterion_levels(human_scores)

        result = score_essay(grade=6, genre=genre, essay_text=essay_text)
        agreement = compute_agreement(result.criteria, human_levels)

        print(f"\n{genre}/{level}: agreement={agreement['rate']:.0%} ({agreement['within_1']}/{agreement['total']})")
        for d in agreement["details"]:
            status = "✓" if d["within_1"] else "✗"
            print(f"  {status} {d['criterion']}: agent={d['agent']} human={d['human']}")

        assert agreement["rate"] >= 0.85, (
            f"Agreement {agreement['rate']:.0%} < 85% threshold for {genre}/{level}"
        )


# ── T032: score_student() integration tests ───────────────────────────────────

class TestScoreStudent:
    """
    Integration tests for score_student() — three-genre aggregate report.
    Uses mock LLM responses derived from golden essay human scores.
    """

    def _make_mock_llm_multi(self, genre_order: list[str]) -> MagicMock:
        """Build a mock LLM whose invoke() returns the correct human scores for
        each genre in sequence (call 0 → genre_order[0], call 1 → genre_order[1], …)."""
        responses = []
        for genre in genre_order:
            _, human_scores = load_golden_case(genre, "proficient")
            responses.append(AIMessage(content=_build_mock_response(human_scores)))
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = responses
        return mock_llm

    def _make_mock_docs(self, genre: str, human_scores: dict) -> list:
        from langchain_core.documents import Document
        return [
            Document(
                page_content=f"Criterion: {c['criterion']}\n",
                metadata={"criterion": c["criterion"], "genre": genre,
                           "grade": 6, "standard": human_scores["standard"]},
            )
            for c in human_scores["criteria"]
        ]

    def test_score_student_returns_student_report(self):
        """score_student returns a StudentReport with three AssessmentScores."""
        from src.scoring.scorer import score_student
        from src.models.result import StudentReport

        genres = ["argumentative", "informative", "narrative"]
        essays = {g: load_golden_case(g, "proficient")[0] for g in genres}
        mock_llm = self._make_mock_llm_multi(genres)

        all_human = {g: load_golden_case(g, "proficient")[1] for g in genres}

        with patch("src.scoring.scorer.retrieve_rubric_chunks") as mock_retrieve:
            mock_retrieve.side_effect = [
                self._make_mock_docs(g, all_human[g]) for g in genres
            ]
            report = score_student(grade=6, essays=essays, llm=mock_llm)

        assert isinstance(report, StudentReport)
        assert report.grade == 6
        assert len(report.assessments) == 3
        assert report.aggregate_total > 0
        assert report.overall_level in ("Advanced", "Proficient", "Developing", "Beginning")

    def test_score_student_aggregate_total_is_sum_of_assessments(self):
        """aggregate_total equals sum of all three individual assessment totals."""
        from src.scoring.scorer import score_student

        genres = ["argumentative", "informative", "narrative"]
        essays = {g: load_golden_case(g, "proficient")[0] for g in genres}
        mock_llm = self._make_mock_llm_multi(genres)

        all_human = {g: load_golden_case(g, "proficient")[1] for g in genres}

        with patch("src.scoring.scorer.retrieve_rubric_chunks") as mock_retrieve:
            mock_retrieve.side_effect = [
                self._make_mock_docs(g, all_human[g]) for g in genres
            ]
            report = score_student(grade=6, essays=essays, llm=mock_llm)

        assert report.aggregate_total == sum(a.total for a in report.assessments)

    def test_score_student_standards_correct_per_genre(self):
        """Each AssessmentScore carries the correct CCSS standard for its genre."""
        from src.scoring.scorer import score_student

        genres = ["argumentative", "informative", "narrative"]
        expected_standards = {"argumentative": "W.6.1", "informative": "W.6.2", "narrative": "W.6.3"}
        essays = {g: load_golden_case(g, "advanced")[0] for g in genres}
        all_human = {g: load_golden_case(g, "advanced")[1] for g in genres}

        responses = []
        for g in genres:
            responses.append(AIMessage(content=_build_mock_response(all_human[g])))
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = responses

        with patch("src.scoring.scorer.retrieve_rubric_chunks") as mock_retrieve:
            mock_retrieve.side_effect = [
                self._make_mock_docs(g, all_human[g]) for g in genres
            ]
            report = score_student(grade=6, essays=essays, llm=mock_llm)

        for assessment in report.assessments:
            assert assessment.standard == expected_standards[assessment.genre], (
                f"Wrong standard for {assessment.genre}: got {assessment.standard}"
            )

    @pytest.mark.parametrize("level,expected_range", [
        ("advanced",   (65, 72)),
        ("proficient", (54, 64)),
        ("developing", (36, 53)),
        ("beginning",  (18, 35)),
    ])
    def test_score_student_overall_level_matches_aggregate_band(self, level, expected_range):
        """Overall performance level maps correctly to the 72-point aggregate scale."""
        from src.scoring.scorer import score_student

        genres = ["argumentative", "informative", "narrative"]
        essays = {g: load_golden_case(g, level)[0] for g in genres}
        all_human = {g: load_golden_case(g, level)[1] for g in genres}

        responses = [AIMessage(content=_build_mock_response(all_human[g])) for g in genres]
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = responses

        with patch("src.scoring.scorer.retrieve_rubric_chunks") as mock_retrieve:
            mock_retrieve.side_effect = [
                self._make_mock_docs(g, all_human[g]) for g in genres
            ]
            report = score_student(grade=6, essays=essays, llm=mock_llm)

        low, high = expected_range
        assert low <= report.aggregate_total <= high, (
            f"aggregate_total={report.aggregate_total} not in [{low},{high}] for level={level}"
        )

    def test_score_student_retrieval_called_once_per_genre(self):
        """Constitution check: retrieve_rubric_chunks called exactly once per genre."""
        from src.scoring.scorer import score_student

        genres = ["argumentative", "informative", "narrative"]
        essays = {g: load_golden_case(g, "proficient")[0] for g in genres}
        all_human = {g: load_golden_case(g, "proficient")[1] for g in genres}

        responses = [AIMessage(content=_build_mock_response(all_human[g])) for g in genres]
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = responses

        with patch("src.scoring.scorer.retrieve_rubric_chunks") as mock_retrieve:
            mock_retrieve.side_effect = [
                self._make_mock_docs(g, all_human[g]) for g in genres
            ]
            score_student(grade=6, essays=essays, llm=mock_llm)

        assert mock_retrieve.call_count == 3
        called_genres = {call.kwargs["genre"] for call in mock_retrieve.call_args_list}
        assert called_genres == {"argumentative", "informative", "narrative"}

    def test_score_student_all_criteria_present_in_each_assessment(self):
        """Every AssessmentScore has exactly 6 CriterionScores."""
        from src.scoring.scorer import score_student

        genres = ["argumentative", "informative", "narrative"]
        essays = {g: load_golden_case(g, "proficient")[0] for g in genres}
        all_human = {g: load_golden_case(g, "proficient")[1] for g in genres}

        responses = [AIMessage(content=_build_mock_response(all_human[g])) for g in genres]
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = responses

        with patch("src.scoring.scorer.retrieve_rubric_chunks") as mock_retrieve:
            mock_retrieve.side_effect = [
                self._make_mock_docs(g, all_human[g]) for g in genres
            ]
            report = score_student(grade=6, essays=essays, llm=mock_llm)

        for assessment in report.assessments:
            assert len(assessment.criteria) == 6, (
                f"{assessment.genre} has {len(assessment.criteria)} criteria, expected 6"
            )
