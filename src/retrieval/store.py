"""
Vector store management using SKLearnVectorStore (pure Python, no native deps).

Collections are stored as pickle files under CHROMA_PERSIST_DIR:
  <persist_dir>/rubrics_grade_6.pkl
  <persist_dir>/rubrics_grade_7.pkl
  <persist_dir>/rubrics_grade_8.pkl
"""
import os
import pickle
import logging
from pathlib import Path

from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME_TEMPLATE = "rubrics_grade_{grade}"


def _persist_path(grade: int, persist_dir: str) -> str:
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return str(Path(persist_dir) / f"rubrics_grade_{grade}.parquet")


def get_persist_dir() -> str:
    return os.getenv("CHROMA_PERSIST_DIR", ".chroma")


def load_or_create_store(
    grade: int,
    embedding: Embeddings,
    persist_dir: str | None = None,
) -> SKLearnVectorStore:
    """
    Load an existing vector store for the given grade, or create an empty one.
    Returns the SKLearnVectorStore instance.
    """
    if persist_dir is None:
        persist_dir = get_persist_dir()

    path = _persist_path(grade, persist_dir)

    if Path(path).exists():
        logger.info("Loading existing vector store for grade %d from %s", grade, path)
        store = SKLearnVectorStore(
            embedding=embedding,
            persist_path=path,
            serializer="parquet",
        )
    else:
        logger.info("Creating new vector store for grade %d at %s", grade, path)
        store = SKLearnVectorStore(
            embedding=embedding,
            persist_path=path,
            serializer="parquet",
        )

    return store


def store_exists(grade: int, persist_dir: str | None = None) -> bool:
    if persist_dir is None:
        persist_dir = get_persist_dir()
    return Path(_persist_path(grade, persist_dir)).exists()
