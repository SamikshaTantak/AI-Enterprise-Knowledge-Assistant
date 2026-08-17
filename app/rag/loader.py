"""
Document loading for the ingestion pipeline.

Scans a source directory for supported files (.pdf, .txt) and loads each
into one or more LangChain `Document` objects with metadata attached
(source filename, file type, and — for PDFs — page number). This metadata
is what later powers source citations in the final answer.

Unsupported or unreadable files are skipped with a logged warning rather
than crashing the whole ingestion run — one bad file should not block
ingestion of everything else.
"""

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class DocumentLoadError(Exception):
    """Raised when the source directory itself is invalid (not a single-file issue)."""


def load_documents(source_dir: str | Path) -> list[Document]:
    """
    Load all supported documents from `source_dir`.

    Args:
        source_dir: Directory containing raw source documents.

    Returns:
        A list of LangChain Document objects, each with metadata:
            - source: original filename (e.g. "remote_work_policy.txt")
            - file_type: "pdf" or "txt"
            - page: page number (PDFs only, 1-indexed)

    Raises:
        DocumentLoadError: if source_dir doesn't exist or contains no
            supported files at all.
    """
    source_path = Path(source_dir)

    if not source_path.exists():
        raise DocumentLoadError(f"Source directory does not exist: {source_path}")
    if not source_path.is_dir():
        raise DocumentLoadError(f"Source path is not a directory: {source_path}")

    all_files = sorted(source_path.iterdir())
    candidate_files = [f for f in all_files if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not candidate_files:
        raise DocumentLoadError(
            f"No supported files (.pdf, .txt) found in {source_path}. "
            f"Found: {[f.name for f in all_files]}"
        )

    documents: list[Document] = []
    skipped: list[str] = []

    for file_path in candidate_files:
        try:
            file_docs = _load_single_file(file_path)
            documents.extend(file_docs)
            logger.info("Loaded %s (%d document(s))", file_path.name, len(file_docs))
        except Exception as e:
            # A single corrupt/unreadable file should not stop the whole run.
            logger.warning("Skipping %s due to load error: %s", file_path.name, e)
            skipped.append(file_path.name)

    if not documents:
        raise DocumentLoadError(
            f"All {len(candidate_files)} candidate file(s) failed to load. "
            f"Skipped: {skipped}"
        )

    if skipped:
        logger.warning("Ingestion completed with %d skipped file(s): %s", len(skipped), skipped)

    return documents


def _load_single_file(file_path: Path) -> list[Document]:
    """Dispatch to the correct loader based on file extension."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        raw_docs = loader.load()
        for i, doc in enumerate(raw_docs):
            doc.metadata["source"] = file_path.name
            doc.metadata["file_type"] = "pdf"
            doc.metadata["page"] = doc.metadata.get("page", i) + 1  # 1-indexed
        return _drop_empty(raw_docs)

    if suffix == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8")
        raw_docs = loader.load()
        for doc in raw_docs:
            doc.metadata["source"] = file_path.name
            doc.metadata["file_type"] = "txt"
        return _drop_empty(raw_docs)

    # Should be unreachable given SUPPORTED_EXTENSIONS filtering upstream.
    raise ValueError(f"Unsupported file extension: {suffix}")


def _drop_empty(docs: list[Document]) -> list[Document]:
    """Filter out pages/files that loaded but contain no usable text."""
    return [d for d in docs if d.page_content and d.page_content.strip()]


if __name__ == "__main__":
    # Quick manual check: `uv run python -m app.rag.loader`
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from app.config.settings import settings

    docs = load_documents("data/raw")
    print(f"\nLoaded {len(docs)} document chunks total.\n")
    for d in docs[:5]:
        preview = d.page_content[:80].replace("\n", " ")
        print(f"  source={d.metadata.get('source')} "
              f"page={d.metadata.get('page', '-')} "
              f"preview='{preview}...'")