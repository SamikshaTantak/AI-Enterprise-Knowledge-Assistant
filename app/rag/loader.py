import logging
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages_text = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append(text.strip())
        else:
            logger.warning(f"  [WARN] {path.name} page {page_num} has no extractable text")
    return "\n\n".join(pages_text).strip()


def load_documents(raw_dir: str) -> list[Document]:

    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw documents directory not found: {raw_dir}")

    documents: list[Document] = []
    all_files = sorted(p for p in raw_path.iterdir() if p.is_file())

    if not all_files:
        logger.warning(f"No files found in {raw_dir}")
        return documents

    for file_path in all_files:
        ext = file_path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            logger.info(f"  [SKIP] {file_path.name} — unsupported extension '{ext}'")
            continue

        try:
            if ext == ".txt":
                text = _load_txt(file_path)
            elif ext == ".pdf":
                text = _load_pdf(file_path)

            if not text:
                logger.warning(f"  [SKIP] {file_path.name} — no extractable text content")
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name,
                        "file_type": ext.lstrip("."),
                        "file_path": str(file_path),
                    },
                )
            )
            logger.info(f"  [OK] Loaded {file_path.name} ({len(text)} chars)")

        except Exception as e:
            logger.error(f"  [FAIL] Could not load {file_path.name}: {e}")
            continue

    return documents


if __name__ == "__main__":
    # Quick manual check: uv run python -m app.rag.loader
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    docs = load_documents("./data/raw")
    print(f"\nLoaded {len(docs)} document(s):")
    for d in docs:
        preview = d.page_content[:100].replace("\n", " ")
        print(f"  - {d.metadata['source']} ({d.metadata['file_type']}): {preview}...")