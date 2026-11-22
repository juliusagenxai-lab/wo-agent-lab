"""
Prompt templates for the essay scoring LLM call.

The scorer calls `build_scoring_prompt(genre, rubric_docs, essay_text)` which
returns a list of LangChain messages ready to send to the LLM.

The LLM is instructed to return a strict JSON array — one object per criterion —
so the output parser can extract scores without ambiguity.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

_SYSTEM_PROMPT = """\
You are an expert writing assessor trained on Common Core State Standards (CCSS) \
for Grades 6–8. Your job is to score a student essay using the provided rubric criteria.

Rules:
1. Score each criterion using ONLY the descriptor text provided — do not use outside knowledge.
2. Assign an integer score from 1 (Beginning) to 4 (Advanced) for each criterion.
3. Write one rationale sentence per criterion that references specific evidence from the essay.
4. Return ONLY a valid JSON array — no prose, no markdown fences, no extra text.
5. When the Level 1 descriptor says "no X" or "absent", and the essay shows only a bare \
   one-sentence trace that is not developed, score it Level 1 — not Level 2 or 3.
6. When in doubt between two adjacent levels, choose the LOWER level. The rubric requires \
   the essay to clearly meet a descriptor before awarding that level.

Output format (array of objects, one per criterion):
[
  {
    "criterion_name": "<exact criterion name from rubric>",
    "level": <integer 1-4>,
    "descriptor": "<the rubric descriptor text for the level you assigned>",
    "rationale": "<one sentence citing specific evidence from the student essay>"
  },
  ...
]
"""

_RUBRIC_SECTION_TEMPLATE = """\
--- RUBRIC CRITERION: {criterion_name} ---
  Level 4 (Advanced):    {l4}
  Level 3 (Proficient):  {l3}
  Level 2 (Developing):  {l2}
  Level 1 (Beginning):   {l1}
"""

_USER_TEMPLATE = """\
Grade {grade} {genre_title} Writing Assessment — Standard {standard}

=== RUBRIC CRITERIA ===
{rubric_text}

=== STUDENT ESSAY ===
{essay_text}

Score each criterion above. Return ONLY the JSON array.
"""


def _doc_to_rubric_section(doc: Document) -> str:
    """Convert a retrieved rubric Document into a formatted rubric section string."""
    content = doc.page_content
    # page_content already has the structured text from chunker; use it directly
    # but also try to extract individual level lines for the template
    lines = content.splitlines()
    criterion_name = doc.metadata.get("criterion", "Unknown")

    level_texts: dict[int, str] = {}
    for line in lines:
        line = line.strip()
        for level, label in [(4, "Advanced"), (3, "Proficient"), (2, "Developing"), (1, "Beginning")]:
            prefix = f"Level {level} ({label}):"
            if line.startswith(prefix):
                level_texts[level] = line[len(prefix):].strip()

    return _RUBRIC_SECTION_TEMPLATE.format(
        criterion_name=criterion_name,
        l4=level_texts.get(4, "No descriptor"),
        l3=level_texts.get(3, "No descriptor"),
        l2=level_texts.get(2, "No descriptor"),
        l1=level_texts.get(1, "No descriptor"),
    )


def build_scoring_prompt(
    grade: int,
    genre: str,
    standard: str,
    rubric_docs: list[Document],
    essay_text: str,
) -> list:
    """
    Build the LLM message list for scoring an essay.

    Args:
        grade:       Student grade level.
        genre:       Writing genre.
        standard:    CCSS standard identifier (e.g. "W.6.1").
        rubric_docs: Retrieved rubric Documents (one per criterion).
        essay_text:  The student's essay text.

    Returns:
        List of LangChain messages [SystemMessage, HumanMessage].
    """
    rubric_text = "\n".join(_doc_to_rubric_section(doc) for doc in rubric_docs)

    user_content = _USER_TEMPLATE.format(
        grade=grade,
        genre_title=genre.capitalize(),
        standard=standard,
        rubric_text=rubric_text,
        essay_text=essay_text.strip(),
    )

    return [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]


# Standard lookup by genre
GENRE_STANDARD: dict[str, dict[int, str]] = {
    "argumentative": {6: "W.6.1", 7: "W.7.1", 8: "W.8.1"},
    "informative":   {6: "W.6.2", 7: "W.7.2", 8: "W.8.2"},
    "narrative":     {6: "W.6.3", 7: "W.7.3", 8: "W.8.3"},
}


def get_standard(grade: int, genre: str) -> str:
    return GENRE_STANDARD.get(genre, {}).get(grade, f"W.{grade}.?")
