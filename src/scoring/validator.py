"""Input validation and output score clamping for the scoring pipeline."""
from dataclasses import dataclass

from src.models.essay import SUPPORTED_GRADES, SUPPORTED_GENRES


@dataclass
class ValidationError:
    message: str
    field: str


def validate_inputs(grade: int, genre: str, essay_text: str) -> list[ValidationError]:
    """
    Validate scoring inputs before retrieval or LLM calls.
    Returns a list of ValidationErrors; empty list means valid.
    """
    errors: list[ValidationError] = []

    if grade not in SUPPORTED_GRADES:
        errors.append(ValidationError(
            field="grade",
            message=f"grade {grade!r} is not supported. Supported grades: {sorted(SUPPORTED_GRADES)}",
        ))

    if genre not in SUPPORTED_GENRES:
        errors.append(ValidationError(
            field="genre",
            message=(
                f"genre {genre!r} is not supported. "
                f"Supported genres: {sorted(SUPPORTED_GENRES)}"
            ),
        ))

    stripped = essay_text.strip() if isinstance(essay_text, str) else ""
    if not stripped:
        errors.append(ValidationError(field="essay_text", message="essay text must not be empty"))
    elif len(stripped) > 5000:
        errors.append(ValidationError(
            field="essay_text",
            message=f"essay text is {len(stripped)} characters; maximum is 5000",
        ))

    return errors


def clamp_level(level: int | float, criterion_name: str = "") -> int:
    """
    Clamp a raw LLM score to the valid 1–4 range.
    Logs a warning if clamping was required.
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        level_int = int(round(float(level)))
    except (TypeError, ValueError):
        logger.warning("Could not parse level %r for criterion %r; defaulting to 1", level, criterion_name)
        return 1

    if level_int < 1 or level_int > 4:
        clamped = max(1, min(4, level_int))
        logger.warning(
            "Score %d out of range for criterion %r; clamped to %d",
            level_int, criterion_name, clamped,
        )
        return clamped

    return level_int
