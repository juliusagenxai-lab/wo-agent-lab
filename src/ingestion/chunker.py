"""
Convert RubricChunks into LangChain Documents ready for embedding.
"""
from langchain_core.documents import Document
from src.ingestion.loader import RubricChunk


def chunks_to_documents(chunks: list[RubricChunk]) -> list[Document]:
    """
    Convert a list of RubricChunks into LangChain Documents.
    Each Document's page_content is the full text of the criterion block;
    metadata carries grade, genre, standard, source, criterion.
    """
    return [
        Document(
            page_content=chunk.to_page_content(),
            metadata=chunk.to_metadata(),
        )
        for chunk in chunks
    ]
