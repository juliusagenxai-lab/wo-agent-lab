"""Result models — scoring output returned to callers."""
from pydantic import BaseModel, field_validator

# Performance scale per-assessment (max 24 points)
_ASSESSMENT_BANDS = [
    (22, 24, "Advanced"),     # 90–100%
    (18, 21, "Proficient"),   # 75–89%
    (12, 17, "Developing"),   # 50–74%
    (6,  11, "Beginning"),    # 25–49%
]

# Performance scale for aggregate report (max 72 points)
_AGGREGATE_BANDS = [
    (65, 72, "Advanced"),
    (54, 64, "Proficient"),
    (36, 53, "Developing"),
    (18, 35, "Beginning"),
]


def _map_level(total: int, bands: list[tuple[int, int, str]]) -> str:
    for low, high, label in bands:
        if low <= total <= high:
            return label
    return "Beginning"


class CriterionScore(BaseModel):
    """The agent's score for a single rubric criterion."""
    criterion_name: str
    level: int              # 1–4
    descriptor: str         # Retrieved rubric text for this level
    rationale: str          # One sentence referencing the student's essay

    @field_validator("level")
    @classmethod
    def level_in_range(cls, v: int) -> int:
        if not (1 <= v <= 4):
            raise ValueError(f"level must be 1–4, got {v}")
        return v


class AssessmentScore(BaseModel):
    """Scoring result for one genre assessment (max 24 points)."""
    grade: int
    genre: str
    standard: str                      # e.g. "W.6.1"
    criteria: list[CriterionScore]
    total: int = 0
    performance_level: str = ""

    def compute_total_and_level(self) -> None:
        """Populate total and performance_level from the criteria list."""
        self.total = sum(c.level for c in self.criteria)
        self.performance_level = _map_level(self.total, _ASSESSMENT_BANDS)


class StudentReport(BaseModel):
    """Full report across all three genre assessments (max 72 points)."""
    grade: int
    assessments: list[AssessmentScore]
    aggregate_total: int = 0
    overall_level: str = ""

    def compute_aggregate(self) -> None:
        """Populate aggregate_total and overall_level from the assessments list."""
        self.aggregate_total = sum(a.total for a in self.assessments)
        self.overall_level = _map_level(self.aggregate_total, _AGGREGATE_BANDS)
