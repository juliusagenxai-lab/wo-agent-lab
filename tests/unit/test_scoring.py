"""
Unit tests for the scoring pipeline.

All LLM calls are mocked — no live endpoint required.
Tests cover: prompt building, output parsing, score clamping, validation gating,
and the full score_essay() flow with a mock LLM.
"""
import json
import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.scoring.prompt import build_scoring_prompt, get_standard, _doc_to_rubric_section
from src.scoring.scorer import _parse_llm_output, _dicts_to_criterion_scores, score_essay


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_rubric_doc(criterion: str, genre: str = "argumentative") -> Document:
    content = (
        f"Grade 6 {genre.capitalize()} Writing – Criterion: {criterion}\n"
        f"  Level 4 (Advanced): Excellent {criterion.lower()} demonstrated.\n"
        f"  Level 3 (Proficient): Adequate {criterion.lower()} demonstrated.\n"
        f"  Level 2 (Developing): Limited {criterion.lower()} demonstrated.\n"
        f"  Level 1 (Beginning): No {criterion.lower()} demonstrated.\n"
    )
    return Document(
        page_content=content,
        metadata={"criterion": criterion, "genre": genre, "grade": 6, "standard": "W.6.1"},
    )


SAMPLE_RUBRIC_DOCS = [
    _make_rubric_doc("Claim"),
    _make_rubric_doc("Evidence"),
    _make_rubric_doc("Reasoning"),
    _make_rubric_doc("Organization"),
    _make_rubric_doc("Counterargument"),
    _make_rubric_doc("Language Conventions"),
]

SAMPLE_ESSAY = "Students should not use phones at school because it distracts from learning."

VALID_LLM_RESPONSE = json.dumps([
    {"criterion_name": "Claim",                "level": 3, "descriptor": "Adequate claim.", "rationale": "Student states a clear position."},
    {"criterion_name": "Evidence",             "level": 2, "descriptor": "Limited evidence.", "rationale": "No specific evidence cited."},
    {"criterion_name": "Reasoning",            "level": 2, "descriptor": "Some reasoning.", "rationale": "Basic connection made."},
    {"criterion_name": "Organization",         "level": 3, "descriptor": "Logical structure.", "rationale": "Paragraphs are organized."},
    {"criterion_name": "Counterargument",      "level": 1, "descriptor": "No opposing view.", "rationale": "No counterargument present."},
    {"criterion_name": "Language Conventions", "level": 3, "descriptor": "Minor errors.", "rationale": "Spelling is mostly correct."},
])


# ── Prompt building ───────────────────────────────────────────────────────────

class TestBuildScoringPrompt:
    def test_returns_two_messages(self):
        msgs = build_scoring_prompt(6, "argumentative", "W.6.1", SAMPLE_RUBRIC_DOCS, SAMPLE_ESSAY)
        assert len(msgs) == 2

    def test_first_message_is_system(self):
        from langchain_core.messages import SystemMessage
        msgs = build_scoring_prompt(6, "argumentative", "W.6.1", SAMPLE_RUBRIC_DOCS, SAMPLE_ESSAY)
        assert isinstance(msgs[0], SystemMessage)

    def test_human_message_contains_essay(self):
        msgs = build_scoring_prompt(6, "argumentative", "W.6.1", SAMPLE_RUBRIC_DOCS, SAMPLE_ESSAY)
        assert SAMPLE_ESSAY in msgs[1].content

    def test_human_message_contains_criterion_names(self):
        msgs = build_scoring_prompt(6, "argumentative", "W.6.1", SAMPLE_RUBRIC_DOCS, SAMPLE_ESSAY)
        content = msgs[1].content
        for criterion in ("Claim", "Evidence", "Reasoning", "Organization"):
            assert criterion in content

    def test_human_message_contains_standard(self):
        msgs = build_scoring_prompt(6, "argumentative", "W.6.1", SAMPLE_RUBRIC_DOCS, SAMPLE_ESSAY)
        assert "W.6.1" in msgs[1].content

    def test_get_standard_returns_correct_values(self):
        assert get_standard(6, "argumentative") == "W.6.1"
        assert get_standard(6, "informative") == "W.6.2"
        assert get_standard(6, "narrative") == "W.6.3"
        assert get_standard(7, "argumentative") == "W.7.1"
        assert get_standard(8, "narrative") == "W.8.3"


class TestDocToRubricSection:
    def test_criterion_name_in_output(self):
        doc = _make_rubric_doc("Claim")
        section = _doc_to_rubric_section(doc)
        assert "Claim" in section

    def test_all_four_levels_in_output(self):
        doc = _make_rubric_doc("Evidence")
        section = _doc_to_rubric_section(doc)
        for label in ("Advanced", "Proficient", "Developing", "Beginning"):
            assert label in section


# ── Output parser ─────────────────────────────────────────────────────────────

class TestParseLLMOutput:
    def test_valid_json_array_parsed(self):
        result = _parse_llm_output(VALID_LLM_RESPONSE, SAMPLE_RUBRIC_DOCS)
        assert isinstance(result, list)
        assert len(result) == 6

    def test_markdown_fences_stripped(self):
        fenced = f"```json\n{VALID_LLM_RESPONSE}\n```"
        result = _parse_llm_output(fenced, SAMPLE_RUBRIC_DOCS)
        assert len(result) == 6

    def test_plain_fences_stripped(self):
        fenced = f"```\n{VALID_LLM_RESPONSE}\n```"
        result = _parse_llm_output(fenced, SAMPLE_RUBRIC_DOCS)
        assert len(result) == 6

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            _parse_llm_output("not json at all", SAMPLE_RUBRIC_DOCS)

    def test_non_array_raises_value_error(self):
        with pytest.raises(ValueError, match="Expected JSON array"):
            _parse_llm_output('{"criterion_name": "Claim", "level": 3}', SAMPLE_RUBRIC_DOCS)


# ── Dicts to CriterionScores ─────────────────────────────────────────────────

class TestDictsToCriterionScores:
    def test_returns_criterion_score_objects(self):
        from src.models.result import CriterionScore
        dicts = json.loads(VALID_LLM_RESPONSE)
        scores = _dicts_to_criterion_scores(dicts, SAMPLE_RUBRIC_DOCS)
        assert all(isinstance(s, CriterionScore) for s in scores)
        assert len(scores) == 6

    def test_out_of_range_level_clamped(self):
        dicts = [{"criterion_name": "Claim", "level": 9, "descriptor": "d", "rationale": "r"}]
        scores = _dicts_to_criterion_scores(dicts, SAMPLE_RUBRIC_DOCS)
        assert scores[0].level == 4

    def test_zero_level_clamped_to_one(self):
        dicts = [{"criterion_name": "Claim", "level": 0, "descriptor": "d", "rationale": "r"}]
        scores = _dicts_to_criterion_scores(dicts, SAMPLE_RUBRIC_DOCS)
        assert scores[0].level == 1


# ── Full score_essay() with mocked LLM ───────────────────────────────────────

def _make_mock_llm(response_text: str):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=response_text)
    return mock_llm


class TestScoreEssay:
    def test_validation_error_raised_before_llm(self):
        """Validation must fail fast — LLM should never be called."""
        mock_llm = _make_mock_llm(VALID_LLM_RESPONSE)
        with pytest.raises(ValueError, match="validation failed"):
            score_essay(grade=5, genre="argumentative", essay_text="text", llm=mock_llm)
        mock_llm.invoke.assert_not_called()

    def test_empty_essay_raises_before_llm(self):
        mock_llm = _make_mock_llm(VALID_LLM_RESPONSE)
        with pytest.raises(ValueError):
            score_essay(grade=6, genre="argumentative", essay_text="", llm=mock_llm)
        mock_llm.invoke.assert_not_called()

    def test_invalid_genre_raises_before_llm(self):
        mock_llm = _make_mock_llm(VALID_LLM_RESPONSE)
        with pytest.raises(ValueError):
            score_essay(grade=6, genre="persuasive", essay_text="text", llm=mock_llm)
        mock_llm.invoke.assert_not_called()

    @patch("src.scoring.scorer.retrieve_rubric_chunks")
    def test_returns_assessment_score(self, mock_retrieve):
        """Full happy path with mocked retriever and LLM."""
        from src.models.result import AssessmentScore
        mock_retrieve.return_value = SAMPLE_RUBRIC_DOCS
        mock_llm = _make_mock_llm(VALID_LLM_RESPONSE)

        result = score_essay(grade=6, genre="argumentative", essay_text=SAMPLE_ESSAY, llm=mock_llm)

        assert isinstance(result, AssessmentScore)
        assert result.grade == 6
        assert result.genre == "argumentative"
        assert result.standard == "W.6.1"
        assert len(result.criteria) == 6
        assert result.total == sum(c.level for c in result.criteria)
        assert result.performance_level in ("Advanced", "Proficient", "Developing", "Beginning")

    @patch("src.scoring.scorer.retrieve_rubric_chunks")
    def test_llm_retries_on_bad_json(self, mock_retrieve):
        """LLM is retried on bad JSON; raises RuntimeError after max retries."""
        mock_retrieve.return_value = SAMPLE_RUBRIC_DOCS
        mock_llm = _make_mock_llm("not valid json")

        with pytest.raises(RuntimeError, match="failed to return valid JSON"):
            score_essay(grade=6, genre="argumentative", essay_text=SAMPLE_ESSAY, llm=mock_llm)

        assert mock_llm.invoke.call_count == 2  # _MAX_RETRIES = 2

    @patch("src.scoring.scorer.retrieve_rubric_chunks")
    def test_llm_invoked_after_retrieval(self, mock_retrieve):
        """Constitution check: retrieval happens before LLM call."""
        mock_retrieve.return_value = SAMPLE_RUBRIC_DOCS
        mock_llm = _make_mock_llm(VALID_LLM_RESPONSE)

        score_essay(grade=6, genre="argumentative", essay_text=SAMPLE_ESSAY, llm=mock_llm)

        mock_retrieve.assert_called_once_with(grade=6, genre="argumentative")
        mock_llm.invoke.assert_called_once()

    @patch("src.scoring.scorer.retrieve_rubric_chunks")
    def test_no_rubric_chunks_raises_runtime_error(self, mock_retrieve):
        """Empty retrieval result raises RuntimeError — never scores from memory."""
        mock_retrieve.return_value = []
        mock_llm = _make_mock_llm(VALID_LLM_RESPONSE)

        with pytest.raises(RuntimeError, match="No rubric chunks retrieved"):
            score_essay(grade=6, genre="argumentative", essay_text=SAMPLE_ESSAY, llm=mock_llm)

        mock_llm.invoke.assert_not_called()

    @patch("src.scoring.scorer.retrieve_rubric_chunks")
    def test_criterion_scores_have_required_fields(self, mock_retrieve):
        mock_retrieve.return_value = SAMPLE_RUBRIC_DOCS
        mock_llm = _make_mock_llm(VALID_LLM_RESPONSE)

        result = score_essay(grade=6, genre="argumentative", essay_text=SAMPLE_ESSAY, llm=mock_llm)

        for cs in result.criteria:
            assert cs.criterion_name
            assert 1 <= cs.level <= 4
            assert cs.descriptor
            assert cs.rationale

    @patch("src.scoring.scorer.retrieve_rubric_chunks")
    def test_out_of_range_llm_scores_clamped(self, mock_retrieve):
        """LLM returning level=9 must be clamped to 4."""
        mock_retrieve.return_value = SAMPLE_RUBRIC_DOCS
        bad_response = json.dumps([
            {"criterion_name": c, "level": 9, "descriptor": "d", "rationale": "r"}
            for c in ("Claim", "Evidence", "Reasoning", "Organization", "Counterargument", "Language Conventions")
        ])
        mock_llm = _make_mock_llm(bad_response)

        result = score_essay(grade=6, genre="argumentative", essay_text=SAMPLE_ESSAY, llm=mock_llm)

        assert all(cs.level == 4 for cs in result.criteria)
