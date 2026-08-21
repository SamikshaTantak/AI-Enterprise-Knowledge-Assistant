import logging
from app.config.settings import settings
from app.rag.chunker import chunk_documents
from app.rag.loader import load_documents
from app.vectorstore.chroma_store import add_chunks, collection_count

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    print("== Step 1/3: Loading documents ==")
    docs = load_documents("./data/raw")
    if not docs:
        print("No documents found in data/raw/ — nothing to ingest.")
        return

    print(f"\n== Step 2/3: Chunking {len(docs)} document(s) ==")
    chunks = chunk_documents(
        docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    print(f"\n== Step 3/3: Embedding + storing {len(chunks)} chunk(s) in Chroma ==")
    stored = add_chunks(chunks)

    total = collection_count()
    print("\n" + "=" * 40)
    print(f"Ingestion complete. Stored {stored} chunk(s) this run.")
    print(f"Collection '{settings.chroma_collection_name}' now has {total} chunk(s) total.")


if __name__ == "__main__":
    main()