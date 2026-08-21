import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """
    Split documents into smaller overlapping chunks.

    Args:
        documents: Documents produced by app.rag.loader.load_documents().
        chunk_size: Max characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of chunked Documents. Each chunk's metadata includes everything
        from the parent document plus `chunk_index` and `chunk_id`.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size})"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks: list[Document] = []

    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        splits = splitter.split_text(doc.page_content)

        if not splits:
            logger.warning(f"  [WARN] {source} produced 0 chunks — skipping")
            continue

        for idx, chunk_text in enumerate(splits):
            chunk_metadata = {
                **doc.metadata,
                "chunk_index": idx,
                "chunk_id": f"{source}::chunk_{idx}",
            }
            all_chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        logger.info(f"  [OK] {source}: {len(splits)} chunk(s)")

    return all_chunks


if __name__ == "__main__":
    # Quick manual check: uv run python -m app.rag.chunker
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from app.config.settings import settings
    from app.rag.loader import load_documents

    docs = load_documents("./data/raw")
    chunks = chunk_documents(
        docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    print(f"\nTotal chunks: {len(chunks)}")
    for c in chunks[:5]:
        preview = c.page_content[:80].replace("\n", " ")
        print(f"  - {c.metadata['chunk_id']} ({len(c.page_content)} chars): {preview}...")
    if len(chunks) > 5:
        print(f"  ... and {len(chunks) - 5} more")