import logging

from langchain_groq import ChatGroq

from app.config.settings import settings
from app.utils.errors import with_retry
from app.workflow.state import GraphState

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an internal Enterprise Knowledge Assistant.

Answer the user's question using ONLY the provided context below.

Rules:
1. Do not use outside knowledge.
2. If the context does not contain enough information to answer confidently,
   clearly say that the available documents do not contain enough information.
3. Do not guess or invent information.
4. Always cite the source document(s) used in your answer using this format:
   [source: filename]

Context:
{context}
"""


def build_context(chunks: list[dict]) -> str:

    parts = []

    for chunk in chunks:
        source = chunk.get("source", "unknown")
        content = chunk.get("content", "")

        parts.append(
            f"[Source: {source}]\n{content}"
        )

    return "\n\n---\n\n".join(parts)


def extract_sources(chunks: list[dict]) -> list[str]:

    sources = []

    for chunk in chunks:
        source = chunk.get("source", "unknown")

        if source not in sources:
            sources.append(source)

    return sources


@with_retry(max_attempts=3, base_delay=1.0, exceptions=(Exception,))
def _invoke_llm(messages: list[tuple[str, str]]):

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0,
    )
    return llm.invoke(messages)


def response_agent(state: GraphState) -> dict:

    query = state["query"]
    chunks = state.get("retrieved_chunks", [])

    logger.info(
        f"[Response Agent] Generating answer using {len(chunks)} chunk(s)"
    )

    if not chunks:
        logger.warning(
            "[Response Agent] No context available for generation"
        )

        return {
            "answer": (
                "I couldn't find enough relevant information in the available "
                "documents to answer this question confidently."
            ),
            "sources": [],
        }

    context = build_context(chunks)
    sources = extract_sources(chunks)

    try:
        messages = [
            (
                "system",
                SYSTEM_PROMPT.format(context=context),
            ),
            ("human", query),
        ]

        response = _invoke_llm(messages)
        answer = str(response.content).strip()

        logger.info(
            "[Response Agent] Answer generated successfully"
        )

        return {
            "answer": answer,
            "sources": sources,
        }

    except Exception as e:
        logger.exception(
            f"[Response Agent] Generation failed: {e}"
        )

        return {
            "answer": (
                "I encountered an error while generating an answer from "
                "the retrieved documents."
            ),
            "sources": sources,
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    test_state: GraphState = {
        "query": "How much can I expense for internet per month?",
        "retrieved_chunks": [
            {
                "chunk_id": "test_chunk_1",
                "source": "expense_reimbursement_policy.txt",
                "content": (
                    "Home internet expenses are not reimbursable. "
                    "Employees may claim eligible business travel expenses."
                ),
            }
        ],
        "answer": "",
        "sources": [],
        "ragas_scores": None,
        "evaluation_summary": None,
    }

    result = response_agent(test_state)

    print("\n" + "=" * 50)
    print("ANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for source in result["sources"]:
        print(f"  - {source}")

    print("=" * 50)