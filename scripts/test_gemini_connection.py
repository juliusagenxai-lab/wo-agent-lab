"""
Quick connection test for the Gemini LLM.

Usage:
    python scripts/test_gemini_connection.py

What it tests:
  1. GOOGLE_API_KEY is set and not a placeholder
  2. Gemini responds to a simple prompt
  3. Gemini returns valid JSON when asked to score a short essay excerpt
     (smoke test only — not a full golden set run)

To run the full golden set validation:
    $env:RAG_TEST_LIVE="1"
    python -m pytest tests/integration/test_score_essay.py -v -s -k "Live"
"""
import json
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


def check_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key or key == "your-google-api-key-here":
        print("❌  GOOGLE_API_KEY is not set in your .env file.")
        print("    Get a free key at: https://aistudio.google.com/app/apikey")
        print("    Then add it to .env:  GOOGLE_API_KEY=your-actual-key")
        sys.exit(1)
    masked = key[:8] + "..." + key[-4:]
    print(f"✅  GOOGLE_API_KEY found: {masked}")
    return key


def test_basic_response(llm) -> None:
    print("\n── Test 1: Basic response ──────────────────────────────────")
    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content="Reply with exactly: OK")])
    # content may be a string or a list of parts depending on the model/version
    raw = response.content
    content = raw if isinstance(raw, str) else " ".join(
        p.get("text", "") if isinstance(p, dict) else str(p) for p in raw
    )
    content = content.strip()
    print(f"   Response: {content!r}")
    assert "OK" in content.upper(), f"Unexpected response: {content}"
    print("✅  Basic response OK")


def test_json_scoring_response(llm) -> None:
    print("\n── Test 2: JSON scoring response ───────────────────────────")
    from langchain_core.messages import SystemMessage, HumanMessage

    system = (
        "You are a writing assessor. Return ONLY a valid JSON array — no markdown, no prose.\n"
        "Each object must have: criterion_name (string), level (integer 1-4), "
        "descriptor (string), rationale (string)."
    )
    user = (
        "Score this essay excerpt on two criteria:\n\n"
        "Essay: 'Students should not use phones at school because it distracts from learning.'\n\n"
        "Criteria:\n"
        "  Claim: 4=compelling, 3=clear, 2=unclear, 1=absent\n"
        "  Evidence: 4=specific, 3=adequate, 2=limited, 1=none\n\n"
        "Return ONLY the JSON array."
    )

    import re

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    # Flatten list-of-parts content (Gemini preview models)
    content = response.content
    if isinstance(content, list):
        content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    raw = content.strip()

    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()

    try:
        data = json.loads(raw)
        assert isinstance(data, list) and len(data) > 0
        for item in data:
            assert "criterion_name" in item
            assert "level" in item
            assert 1 <= int(item["level"]) <= 4
        print(f"   Parsed {len(data)} criterion score(s):")
        for item in data:
            print(f"     {item['criterion_name']}: level={item['level']}")
        print("✅  JSON scoring response OK")
    except (json.JSONDecodeError, AssertionError, KeyError) as exc:
        print(f"❌  JSON response invalid: {exc}")
        print(f"   Raw response:\n{raw}")
        sys.exit(1)


def test_full_score_essay(llm) -> None:
    print("\n── Test 3: Full score_essay() pipeline ─────────────────────")
    from src.scoring.scorer import score_essay

    essay = Path("tests/golden/grade_6_argumentative/essay_proficient.txt").read_text(encoding="utf-8")

    try:
        result = score_essay(grade=6, genre="argumentative", essay_text=essay, llm=llm)
        print(f"   total={result.total}/24  level={result.performance_level}")
        print(f"   standard={result.standard}")
        print(f"   criteria scored: {len(result.criteria)}/6")
        for cs in result.criteria:
            print(f"     {cs.criterion_name:25s} → {cs.level}  ({cs.rationale[:60]}...)")
        print("✅  Full score_essay() pipeline OK")
    except Exception as exc:
        print(f"❌  score_essay() failed: {exc}")
        sys.exit(1)


def main():
    print("═" * 60)
    print("  Gemini LLM Connection Test")
    print("  RAG Writing Assessment Agent")
    print("═" * 60)

    model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    check_api_key()

    print(f"\n── Connecting to Gemini model: {model} ──────────────────")
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
        max_output_tokens=2048,
    )
    print("✅  LLM client built")

    test_basic_response(llm)
    test_json_scoring_response(llm)
    test_full_score_essay(llm)

    print("\n" + "═" * 60)
    print("  All tests passed — Gemini is connected and scoring works!")
    print("═" * 60)
    print("\nNext step — run the full golden set validation:")
    print('  $env:RAG_TEST_LIVE="1"')
    print("  python -m pytest tests/integration/test_score_essay.py -v -s -k Live")


if __name__ == "__main__":
    main()
