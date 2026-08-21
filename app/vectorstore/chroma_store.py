import logging
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.config.settings import settings
from app.rag.embeddings import get_embedding_model
from app.utils.errors import VectorStoreError, with_retry

logger = logging.getLogger(__name__)


def get_vectorstore() -> Chroma:
    try:
        return Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=get_embedding_model(),
            persist_directory=settings.chroma_persist_dir,
        )
    except Exception as e:
        raise VectorStoreError(f"Could not open ChromaDB collection: {e}") from e


def add_chunks(chunks: list[Document]) -> int:

    if not chunks:
        logger.warning("No chunks provided — nothing to store")
        return 0

    try:
        store = get_vectorstore()
        ids = [c.metadata["chunk_id"] for c in chunks]
        store.add_documents(documents=chunks, ids=ids)
    except VectorStoreError:
        raise
    except Exception as e:
        raise VectorStoreError(f"Failed to store chunks in ChromaDB: {e}") from e

    logger.info(
        f"  [OK] Stored {len(chunks)} chunk(s) in Chroma collection "
        f"'{settings.chroma_collection_name}'"
    )
    return len(chunks)


@with_retry(max_attempts=3, base_delay=1.0, exceptions=(VectorStoreError,))
def similarity_search(query: str, k: int | None = None) -> list[Document]:
  
    try:
        store = get_vectorstore()
        top_k = k if k is not None else settings.retrieval_top_k
        return store.similarity_search(query, k=top_k)
    except VectorStoreError:
        raise
    except Exception as e:
        raise VectorStoreError(f"Similarity search failed: {e}") from e


def collection_count() -> int:
    try:
        store = get_vectorstore()
        return store._collection.count()
    except Exception as e:
        logger.warning(f"[chroma_store] collection_count failed, returning 0: {e}")
        return 0