"""
Retriever — loads the grade-specific vector store and retrieves rubric chunks
filtered by genre.

Post-filter approach: SKLearnVectorStore does not support native metadata
filtering, so we over-fetch (k*3) and filter by genre in Python.
"""
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Number of rubric chunks to retrieve per query (one per criterion)
_DEFAULT_K = 6

# Module-level singleton — loaded once, reused across all retrieve_rubric_chunks calls.
# This also avoids the tqdm/Windows stderr crash that occurs when the HuggingFace
# model loader writes progress bars to a captured stderr (pytest -s).
_EMBEDDING = None


def _load_embedding():
    global _EMBEDDING
    if _EMBEDDING is not None:
        return _EMBEDDING

    # Suppress tqdm progress bars during model loading to avoid Windows/pytest
    # OSError: [Errno 22] Invalid argument when stderr is captured.
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from langchain_community.embeddings import SentenceTransformerEmbeddings  # type: ignore[import-untyped]
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    _EMBEDDING = SentenceTransformerEmbeddings(model_name=model_name)
    return _EMBEDDING


def _load_store(grade: int, persist_dir: str | None = None):
    """Load and return the SKLearnVectorStore for a given grade."""
    from langchain_community.vectorstores import SKLearnVectorStore

    if persist_dir is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", ".chroma")

    persist_path = str(Path(persist_dir) / f"rubrics_grade_{grade}.parquet")

    if not Path(persist_path).exists():
        raise FileNotFoundError(
            f"Vector store not found for grade {grade} at {persist_path}. "
            f"Run: python -m src.ingestion.ingest --grades {grade}"
        )

    embedding = _load_embedding()
    logger.info("Loading vector store for grade %d from %s", grade, persist_path)

    return SKLearnVectorStore(
        embedding=embedding,
        persist_path=persist_path,
        serializer="parquet",
    )


def retrieve_rubric_chunks(
    grade: int,
    genre: str,
    k: int = _DEFAULT_K,
    persist_dir: str | None = None,
) -> list:
    """
    Retrieve rubric Document chunks for the given grade and genre.

    Over-fetches k*3 candidates by similarity, post-filters to the requested
    genre, de-duplicates by criterion, and returns at most k documents.

    Args:
        grade:       Student grade level (6, 7, or 8).
        genre:       Writing genre ("argumentative", "informative", or "narrative").
        k:           Maximum number of chunks to return (default 6).
        persist_dir: Override the CHROMA_PERSIST_DIR env var.

    Returns:
        List of LangChain Document objects, one per unique rubric criterion.

    Raises:
        FileNotFoundError: If the vector store for the given grade has not been ingested.
    """
    store = _load_store(grade=grade, persist_dir=persist_dir)
    query = f"Grade {grade} {genre} writing rubric scoring criteria"

    # Fetch more than needed so genre filtering has candidates to work with
    candidates = store.similarity_search(query, k=k * 3)

    # Post-filter by genre metadata
    filtered = [d for d in candidates if d.metadata.get("genre") == genre]

    # De-duplicate by criterion name, preserving similarity order
    seen: set[str] = set()
    unique: list = []
    for doc in filtered:
        crit = doc.metadata.get("criterion", "")
        if crit not in seen:
            seen.add(crit)
            unique.append(doc)

    result = unique[:k]
    logger.info("Retrieved %d chunks for grade=%d genre=%s", len(result), grade, genre)
    return result
