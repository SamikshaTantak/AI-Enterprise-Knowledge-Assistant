import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.utils.errors import MCPError

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SERVER_FILE = (
    PROJECT_ROOT
    / "app"
    / "mcp"
    / "document_server.py"
)

MCP_CALL_TIMEOUT_SECONDS = 20


async def _call_mcp_tool(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> Any:
    """
    Start the Enterprise Documents MCP Server and call a tool.

    Args:
        tool_name: Name of the MCP tool to call.
        arguments: Arguments passed to the tool.

    Returns:
        The structured result returned by the MCP tool.
    """

    server_params = StdioServerParameters(
        command="uv",
        args=[
            "run",
            "python",
            str(SERVER_FILE),
        ],
        cwd=str(PROJECT_ROOT),
    )

    async with stdio_client(
        server_params
    ) as (read, write):

        async with ClientSession(
            read,
            write,
        ) as session:

            await session.initialize()

            result = await session.call_tool(
                tool_name,
                arguments=arguments or {},
            )

            if result.is_error:
                raise RuntimeError(
                    f"MCP tool '{tool_name}' failed: "
                    f"{result.content}"
                )

            # Prefer structured output.
            if result.structured_content is not None:
                return result.structured_content

            # Fallback: parse text content.
            if result.content:
                text = getattr(
                    result.content[0],
                    "text",
                    None,
                )

                if text:
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return text

            return None


def _run_mcp_tool_sync(tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
    try:
        return asyncio.run(
            asyncio.wait_for(
                _call_mcp_tool(tool_name, arguments),
                timeout=MCP_CALL_TIMEOUT_SECONDS,
            )
        )
    except TimeoutError as e:
        raise MCPError(
            f"MCP tool '{tool_name}' timed out after "
            f"{MCP_CALL_TIMEOUT_SECONDS}s"
        ) from e
    except Exception as e:
        raise MCPError(f"MCP tool '{tool_name}' failed: {e}") from e


def list_enterprise_documents() -> list[str]:
    result = _run_mcp_tool_sync("list_enterprise_documents")

    if isinstance(result, dict):
        return result.get("result", [])

    return result or []


def search_enterprise_documents(
    query: str,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """
    Search enterprise documents using the MCP server.

    Args:
        query: User search query.
        max_results: Maximum number of results.

    Returns:
        Matching enterprise document results.

    Raises:
        MCPError: if the server is unreachable or times out.
                   NOTE: retriever_agent.py already catches this broadly
                   and returns [] — behavior is unchanged, only the
                   logged error detail improves, and calls now reliably
                   fail fast instead of hanging.
    """
    result = _run_mcp_tool_sync(
        "search_enterprise_documents",
        {
            "query": query,
            "max_results": max_results,
        },
    )

    if isinstance(result, dict):
        return result.get("result", [])

    return result or []


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("\n" + "=" * 60)
    print("MCP CLIENT TEST")
    print("=" * 60)

    print("\nAvailable Enterprise Documents:")

    documents = list_enterprise_documents()

    for document in documents:
        print(f"  - {document}")

    query = "internet expense reimbursement"

    print(f"\nSearching for: '{query}'")

    results = search_enterprise_documents(
        query=query,
        max_results=5,
    )

    print("\nSearch Results:")

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n{index}. Source: "
            f"{result.get('source')}"
        )

        print(
            f"   Match Score: "
            f"{result.get('match_score')}"
        )

        content = result.get(
            "content",
            "",
        )

        print(
            f"   Content Preview: "
            f"{content[:200]}"
        )

    print("\n" + "=" * 60)