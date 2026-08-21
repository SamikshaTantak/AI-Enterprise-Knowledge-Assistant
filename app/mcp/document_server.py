from pathlib import Path
from mcp.server import MCPServer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "raw"

mcp = MCPServer(
    "Enterprise Documents Server",
    instructions=(
        "Use the available tools to search enterprise documents "
        "and retrieve relevant information."
    ),
)


@mcp.tool()
def list_enterprise_documents() -> list[str]:
    """
    List all available enterprise documents.
    """

    if not DOCUMENTS_DIR.exists():
        return []

    documents = []

    for file_path in DOCUMENTS_DIR.iterdir():
        if file_path.is_file():
            documents.append(file_path.name)

    return sorted(documents)


@mcp.tool()
def search_enterprise_documents(
    query: str,
    max_results: int = 5,
) -> list[dict]:
    """
    Search enterprise documents for text relevant to a query.

    Args:
        query: The text to search for.
        max_results: Maximum number of matching results to return.

    Returns:
        A list containing matching document names and content snippets.
    """

    if not DOCUMENTS_DIR.exists():
        return []

    query_terms = [
        term.lower()
        for term in query.split()
        if len(term.strip()) > 2
    ]

    if not query_terms:
        return []

    results = []

    supported_extensions = {
        ".txt",
        ".md",
    }

    for file_path in DOCUMENTS_DIR.iterdir():

        if (
            not file_path.is_file()
            or file_path.suffix.lower() not in supported_extensions
        ):
            continue

        try:
            content = file_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except OSError:
            continue

        content_lower = content.lower()

        matches = sum(
            term in content_lower
            for term in query_terms
        )

        if matches == 0:
            continue

        results.append(
            {
                "source": file_path.name,
                "content": content[:2000],
                "match_score": matches,
            }
        )

    results.sort(
        key=lambda item: item["match_score"],
        reverse=True,
    )

    return results[:max_results]


if __name__ == "__main__":
    mcp.run(transport="stdio")