"""
Load rubric Markdown files from data/rubrics/<grade>/ and parse them into
structured chunks with metadata.

Expected file format (YAML front-matter + Markdown body):
---
grade: 6
genre: argumentative
standard: W.6.1
source: "CCSS/CA CCSS Grade 6 Writing Standards"
---
# Grade 6 Argumentative Writing Rubric
...
## Criterion: <name>
| Level | Score | Descriptor | ...
"""
import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class RubricChunk:
    """A single criterion block parsed from a rubric file."""
    grade: int
    genre: str
    standard: str
    source: str
    criterion: str
    level_descriptors: dict[int, str]   # {4: "...", 3: "...", 2: "...", 1: "..."}

    def to_page_content(self) -> str:
        """Return the chunk as plain text for embedding."""
        lines = [f"Grade {self.grade} {self.genre.capitalize()} Writing – Criterion: {self.criterion}"]
        level_labels = {4: "Advanced", 3: "Proficient", 2: "Developing", 1: "Beginning"}
        for level in sorted(self.level_descriptors, reverse=True):
            lines.append(f"  Level {level} ({level_labels[level]}): {self.level_descriptors[level]}")
        return "\n".join(lines)

    def to_metadata(self) -> dict:
        return {
            "grade": self.grade,
            "genre": self.genre,
            "standard": self.standard,
            "source": self.source,
            "criterion": self.criterion,
        }


_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_CRITERION_HEADER_RE = re.compile(r"^## Criterion:\s*(.+)$", re.MULTILINE)
# Matches table rows: | Level name | Score | Descriptor text |
_TABLE_ROW_RE = re.compile(
    r"^\|\s*(Advanced|Proficient|Developing|Beginning)\s*\|\s*(\d)\s*\|\s*(.+?)\s*\|",
    re.MULTILINE | re.IGNORECASE,
)
_LEVEL_MAP = {"advanced": 4, "proficient": 3, "developing": 2, "beginning": 1}


def _parse_front_matter(text: str) -> dict:
    """Extract simple key: value pairs from YAML-style front matter."""
    meta: dict = {}
    for line in text.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"')
    return meta


def load_rubric_file(path: Path) -> list[RubricChunk]:
    """Parse a rubric Markdown file and return one RubricChunk per criterion."""
    text = path.read_text(encoding="utf-8")

    # Parse front matter
    fm_match = _FRONT_MATTER_RE.match(text)
    if not fm_match:
        raise ValueError(f"No YAML front matter found in {path}")
    meta = _parse_front_matter(fm_match.group(1))

    grade = int(meta.get("grade", 0))
    genre = meta.get("genre", "")
    standard = meta.get("standard", "")
    source = meta.get("source", str(path))

    body = text[fm_match.end():]

    # Split body into per-criterion sections.
    # re.split with a capturing group produces:
    #   [preamble, name1, section1, name2, section2, ...]
    # So parts[1::2] = names, parts[2::2] = sections.
    split_parts = _CRITERION_HEADER_RE.split(body)
    criterion_names = split_parts[1::2]
    criterion_sections = split_parts[2::2]

    chunks: list[RubricChunk] = []
    for name, section in zip(criterion_names, criterion_sections):
        descriptors: dict[int, str] = {}
        for row in _TABLE_ROW_RE.finditer(section):
            level_label = row.group(1).lower()
            descriptor = row.group(3).strip()
            level = _LEVEL_MAP.get(level_label)
            if level is not None:
                descriptors[level] = descriptor

        if descriptors:
            chunks.append(RubricChunk(
                grade=grade,
                genre=genre,
                standard=standard,
                source=source,
                criterion=name.strip(),
                level_descriptors=descriptors,
            ))

    return chunks


def load_grade_rubrics(grade: int, rubrics_dir: str = "data/rubrics") -> list[RubricChunk]:
    """Load all three genre rubric files for a given grade."""
    grade_dir = Path(rubrics_dir) / f"grade_{grade}"
    if not grade_dir.exists():
        raise FileNotFoundError(f"Rubric directory not found: {grade_dir}")

    all_chunks: list[RubricChunk] = []
    for md_file in sorted(grade_dir.glob("*.md")):
        chunks = load_rubric_file(md_file)
        all_chunks.extend(chunks)

    return all_chunks
