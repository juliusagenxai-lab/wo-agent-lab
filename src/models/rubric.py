"""Rubric data models — criteria descriptors retrieved from the knowledge base."""
from pydantic import BaseModel


class RubricCriterion(BaseModel):
    """A single scoring dimension with descriptors for each of the four score levels."""
    name: str
    standard: str          # e.g. "W.6.1"
    grade: int
    genre: str
    source: str
    # Maps score level (1–4) to the descriptor text for that level
    level_descriptors: dict[int, str]

    def descriptor_for(self, level: int) -> str:
        """Return the descriptor text for a given score level (1–4)."""
        return self.level_descriptors.get(level, f"No descriptor for level {level}")

    def all_descriptors_text(self) -> str:
        """Return all level descriptors as a formatted string for prompt injection."""
        lines = [f"Criterion: {self.name}"]
        for level in sorted(self.level_descriptors):
            label = {4: "Advanced", 3: "Proficient", 2: "Developing", 1: "Beginning"}[level]
            lines.append(f"  {level} – {label}: {self.level_descriptors[level]}")
        return "\n".join(lines)
