"""Unit tests for Pydantic data models."""
import pytest
from pydantic import ValidationError as PydanticValidationError

from src.models.essay import Essay
from src.models.result import AssessmentScore, CriterionScore, StudentReport


# ── Essay model ──────────────────────────────────────────────────────────────

class TestEssayModel:
    def test_valid_essay(self):
        e = Essay(grade=6, genre="argumentative", text="Some essay text.")
        assert e.grade == 6
        assert e.genre == "argumentative"
        assert e.text == "Some essay text."

    def test_text_is_stripped(self):
        e = Essay(grade=6, genre="narrative", text="  Hello world.  ")
        assert e.text == "Hello world."

    def test_invalid_grade(self):
        with pytest.raises(PydanticValidationError):
            Essay(grade=5, genre="argumentative", text="text")

    def test_invalid_genre(self):
        with pytest.raises(PydanticValidationError):
            Essay(grade=6, genre="persuasive", text="text")

    def test_empty_text_raises(self):
        with pytest.raises(PydanticValidationError):
            Essay(grade=6, genre="narrative", text="   ")

    def test_text_too_long_raises(self):
        with pytest.raises(PydanticValidationError):
            Essay(grade=6, genre="narrative", text="x" * 5001)

    def test_all_supported_grades(self):
        for grade in (6, 7, 8):
            e = Essay(grade=grade, genre="informative", text="hello")
            assert e.grade == grade

    def test_all_supported_genres(self):
        for genre in ("argumentative", "informative", "narrative"):
            e = Essay(grade=6, genre=genre, text="hello")
            assert e.genre == genre


# ── CriterionScore model ─────────────────────────────────────────────────────

class TestCriterionScore:
    def test_valid_criterion_score(self):
        cs = CriterionScore(
            criterion_name="Claim",
            level=3,
            descriptor="Clear claim.",
            rationale="The student states a clear position.",
        )
        assert cs.level == 3

    def test_level_out_of_range_raises(self):
        with pytest.raises(PydanticValidationError):
            CriterionScore(
                criterion_name="Claim", level=5,
                descriptor="x", rationale="y",
            )

    def test_level_zero_raises(self):
        with pytest.raises(PydanticValidationError):
            CriterionScore(
                criterion_name="Claim", level=0,
                descriptor="x", rationale="y",
            )


# ── AssessmentScore.compute_total_and_level ───────────────────────────────────

def _make_assessment(levels: list[int], genre: str = "argumentative") -> AssessmentScore:
    criteria = [
        CriterionScore(
            criterion_name=f"C{i}", level=lv,
            descriptor="desc", rationale="rationale",
        )
        for i, lv in enumerate(levels)
    ]
    a = AssessmentScore(grade=6, genre=genre, standard="W.6.1", criteria=criteria)
    a.compute_total_and_level()
    return a


class TestAssessmentScoreLevel:
    @pytest.mark.parametrize("levels,expected_level", [
        ([4, 4, 4, 4, 4, 4], "Advanced"),      # 24/24 = 100%
        ([4, 4, 4, 4, 4, 2], "Advanced"),      # 22/24 = 92%
        ([4, 4, 3, 3, 3, 3], "Proficient"),    # 20/24 = 83%
        ([3, 3, 3, 3, 3, 3], "Proficient"),    # 18/24 = 75%
        ([3, 3, 2, 2, 2, 2], "Developing"),    # 14/24 = 58%
        ([2, 2, 2, 2, 2, 2], "Developing"),    # 12/24 = 50%
        ([2, 2, 1, 1, 2, 2], "Beginning"),     # 10/24 = 42%
        ([1, 1, 1, 1, 1, 1], "Beginning"),     # 6/24  = 25%
    ])
    def test_performance_level(self, levels, expected_level):
        a = _make_assessment(levels)
        assert a.performance_level == expected_level

    def test_total_is_sum_of_levels(self):
        a = _make_assessment([4, 3, 3, 4, 2, 2])
        assert a.total == 18


# ── StudentReport.compute_aggregate ──────────────────────────────────────────

def _make_report(totals: list[int]) -> StudentReport:
    assessments = []
    genres = ["argumentative", "informative", "narrative"]
    stds = ["W.6.1", "W.6.2", "W.6.3"]
    for i, total in enumerate(totals):
        # Build criteria that sum to the desired total
        per = total // 6
        remainder = total % 6
        levels = [per] * 6
        for j in range(remainder):
            levels[j] += 1
        criteria = [
            CriterionScore(
                criterion_name=f"C{j}", level=lv,
                descriptor="d", rationale="r",
            )
            for j, lv in enumerate(levels)
        ]
        a = AssessmentScore(
            grade=6, genre=genres[i], standard=stds[i],
            criteria=criteria, total=total,
            performance_level="Proficient",
        )
        assessments.append(a)
    report = StudentReport(grade=6, assessments=assessments)
    report.compute_aggregate()
    return report


class TestStudentReport:
    @pytest.mark.parametrize("totals,expected_level", [
        ([22, 22, 21], "Advanced"),     # 65/72 = 90%
        ([20, 18, 19], "Proficient"),   # 57/72 = 79%
        ([15, 14, 15], "Developing"),   # 44/72 = 61%
        ([10, 10, 10], "Beginning"),    # 30/72 = 42%
    ])
    def test_aggregate_level(self, totals, expected_level):
        report = _make_report(totals)
        assert report.overall_level == expected_level

    def test_aggregate_total_is_sum(self):
        report = _make_report([20, 18, 19])
        assert report.aggregate_total == 57
