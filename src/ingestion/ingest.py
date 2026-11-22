"""
Ingestion entry point.

Usage:
    python -m src.ingestion.ingest [--grades 6 7 8] [--rubrics-dir data/rubrics]

Loads rubric Markdown files, embeds them, and persists to the vector store.
Idempotent: existing stores are overwritten (safe to re-run after rubric updates).
"""
import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_embedding():
    from langchain_community.embeddings import SentenceTransformerEmbeddings  # type: ignore[import-untyped]
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformerEmbeddings(model_name=model_name)


def ingest_grade(grade: int, rubrics_dir: str, persist_dir: str, embedding) -> int:
    """
    Load, chunk, embed, and persist rubric documents for a single grade.
    Returns the number of documents ingested.
    """
    from src.ingestion.loader import load_grade_rubrics
    from src.ingestion.chunker import chunks_to_documents
    from langchain_community.vectorstores import SKLearnVectorStore
    from pathlib import Path

    logger.info("Ingesting grade %d rubrics from %s", grade, rubrics_dir)
    chunks = load_grade_rubrics(grade, rubrics_dir)
    documents = chunks_to_documents(chunks)

    if not documents:
        logger.warning("No documents found for grade %d — skipping", grade)
        return 0

    persist_path = str(Path(persist_dir) / f"rubrics_grade_{grade}.parquet")
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Embedding %d documents for grade %d → %s", len(documents), grade, persist_path)
    store = SKLearnVectorStore.from_documents(
        documents=documents,
        embedding=embedding,
        persist_path=persist_path,
        serializer="parquet",
    )
    store.persist()
    logger.info("Grade %d ingestion complete — %d documents stored", grade, len(documents))
    return len(documents)


def main(grades: list[int] | None = None, rubrics_dir: str = "data/rubrics") -> None:
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", ".chroma")

    if grades is None:
        grades = [6, 7, 8]

    embedding = _build_embedding()
    total = 0

    for grade in grades:
        try:
            count = ingest_grade(grade, rubrics_dir, persist_dir, embedding)
            total += count
        except FileNotFoundError as exc:
            logger.warning("Skipping grade %d: %s", grade, exc)

    logger.info("Ingestion complete — %d total documents across grades %s", total, grades)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest rubric documents into the vector store")
    parser.add_argument(
        "--grades", nargs="+", type=int, default=[6, 7, 8],
        help="Grades to ingest (default: 6 7 8)",
    )
    parser.add_argument(
        "--rubrics-dir", default="data/rubrics",
        help="Path to rubric Markdown files (default: data/rubrics)",
    )
    args = parser.parse_args()
    main(grades=args.grades, rubrics_dir=args.rubrics_dir)
