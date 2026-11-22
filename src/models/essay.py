"""Essay input model — the student submission to be scored."""
from typing import Literal
from pydantic import BaseModel, field_validator

SUPPORTED_GRADES = (6, 7, 8)
SUPPORTED_GENRES = ("argumentative", "informative", "narrative")


class Essay(BaseModel):
    grade: int
    genre: Literal["argumentative", "informative", "narrative"]
    text: str

    @field_validator("grade")
    @classmethod
    def grade_must_be_supported(cls, v: int) -> int:
        if v not in SUPPORTED_GRADES:
            raise ValueError(f"grade must be one of {SUPPORTED_GRADES}, got {v}")
        return v

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("essay text must not be empty")
        if len(stripped) > 5000:
            raise ValueError("essay text exceeds maximum length of 5000 characters")
        return stripped
