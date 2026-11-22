"""
CLI entry point for the RAG Writing Assessment Agent.

Usage — single essay:
    python -m src.cli --grade 6 --genre argumentative --file essay.txt

Usage — all three genres (full student report):
    python -m src.cli --all --grade 6 \
        --argumentative arg.txt \
        --informative   inf.txt \
        --narrative     nar.txt
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,          # suppress INFO noise in CLI output
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return p.read_text(encoding="utf-8")


def cmd_score_single(args) -> None:
    """Score a single essay and print the AssessmentScore as JSON."""
    from src.scoring.scorer import score_essay

    essay_text = _read_file(args.file)
    try:
        result = score_essay(grade=args.grade, genre=args.genre, essay_text=essay_text)
        print(result.model_dump_json(indent=2))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_score_all(args) -> None:
    """Score all three genre essays and print the StudentReport as JSON."""
    from src.scoring.scorer import score_student

    essays: dict[str, str] = {}
    for genre in ("argumentative", "informative", "narrative"):
        path = getattr(args, genre, None)
        if not path:
            print(f"Error: --{genre} <file> is required with --all", file=sys.stderr)
            sys.exit(1)
        essays[genre] = _read_file(path)

    try:
        report = score_student(grade=args.grade, essays=essays)
        print(report.model_dump_json(indent=2))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Writing Assessment Agent — score student essays against CCSS rubrics",
    )
    parser.add_argument("--grade", type=int, required=True, help="Grade level (6, 7, or 8)")

    sub = parser.add_subparsers(dest="command")

    # Single essay command
    single = sub.add_parser("score", help="Score a single essay")
    single.add_argument("--genre", required=True,
                        choices=["argumentative", "informative", "narrative"],
                        help="Writing genre")
    single.add_argument("--file", required=True, help="Path to the essay text file")

    # All-three command
    all_cmd = sub.add_parser("score-all", help="Score all three genre essays for one student")
    all_cmd.add_argument("--argumentative", required=True, metavar="FILE")
    all_cmd.add_argument("--informative", required=True, metavar="FILE")
    all_cmd.add_argument("--narrative", required=True, metavar="FILE")

    args = parser.parse_args()

    if args.command == "score":
        cmd_score_single(args)
    elif args.command == "score-all":
        cmd_score_all(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
