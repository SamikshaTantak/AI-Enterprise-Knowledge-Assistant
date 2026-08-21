import logging

from app.mcp.document_client import search_enterprise_documents
from app.vectorstore.chroma_store import similarity_search
from app.workflow.state import ChunkResult, GraphState


logger = logging.getLogger(__name__)


def _get_chroma_chunks(query: str) -> list[ChunkResult]:

    try:
        docs = similarity_search(query)

        chunks: list[ChunkResult] = [
            {
                "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                "source": doc.metadata.get("source", "unknown"),
                "content": doc.page_content,
            }
            for doc in docs
        ]

        logger.info(
            f"[Retriever Agent] ChromaDB retrieved "
            f"{len(chunks)} chunk(s)"
        )

        return chunks

    except Exception as e:
        logger.error(
            f"[Retriever Agent] ChromaDB retrieval failed: {e}"
        )

        return []


def _get_mcp_chunks(query: str) -> list[ChunkResult]:
  
    try:
        results = search_enterprise_documents(
            query=query,
            max_results=5,
        )

        chunks: list[ChunkResult] = []

        for index, result in enumerate(results, start=1):
            source = result.get("source", "unknown")
            content = result.get("content", "")

            if not content:
                continue

            chunks.append(
                {
                    "chunk_id": f"mcp_{source}_{index}",
                    "source": source,
                    "content": content,
                }
            )

        logger.info(
            f"[Retriever Agent] MCP retrieved "
            f"{len(chunks)} result(s)"
        )

        return chunks

    except Exception as e:
        logger.error(
            f"[Retriever Agent] MCP retrieval failed: {e}"
        )

        return []


def _merge_chunks(
    chroma_chunks: list[ChunkResult],
    mcp_chunks: list[ChunkResult],
) -> list[ChunkResult]:
  
    merged_chunks = list(chroma_chunks)

    existing_sources = {
        chunk["source"]
        for chunk in chroma_chunks
    }

    for chunk in mcp_chunks:
        source = chunk["source"]

        if source in existing_sources:
            logger.info(
                f"[Retriever Agent] Skipping duplicate MCP source: "
                f"{source}"
            )
            continue

        merged_chunks.append(chunk)
        existing_sources.add(source)

    logger.info(
        f"[Retriever Agent] Total merged chunks: "
        f"{len(merged_chunks)}"
    )

    return merged_chunks


def retriever_agent(state: GraphState) -> dict:
    """
    LangGraph Retriever node.

    Uses both:
    - ChromaDB semantic retrieval
    - MCP Enterprise Documents search
    """

    query = state["query"]

    logger.info(
        f"[Retriever Agent] Searching for: {query!r}"
    )

    chroma_chunks = _get_chroma_chunks(query)

    mcp_chunks = _get_mcp_chunks(query)

    merged_chunks = _merge_chunks(
        chroma_chunks=chroma_chunks,
        mcp_chunks=mcp_chunks,
    )

    logger.info(
        f"[Retriever Agent] Retrieval completed with "
        f"{len(merged_chunks)} chunk(s)"
    )

    return {
        "retrieved_chunks": merged_chunks
    }


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    test_state: GraphState = {
        "query": "How much can I expense for internet per month?",
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "ragas_scores": None,
        "evaluation_summary": None,
    }

    result = retriever_agent(test_state)

    print("\nRetrieved Chunks:")

    for chunk in result["retrieved_chunks"]:
        print(
            f"\n- {chunk['chunk_id']}"
        )
        print(
            f"  Source: {chunk['source']}"
        )
        print(
            f"  Preview: {chunk['content'][:150]}..."
        )