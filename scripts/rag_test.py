import sys
from langchain_groq import ChatGroq
from app.config.settings import settings
from app.vectorstore.chroma_store import similarity_search

SYSTEM_PROMPT = """You are an internal Enterprise Knowledge Assistant.
Answer the user's question using ONLY the provided context below.
If the context does not contain enough information to answer confidently,
say so explicitly instead of guessing.

Always cite which source document(s) you used in your answer, using the
format [source: filename].

Context:
{context}
"""

def build_context(chunks) -> str:
    """Format retrieved chunks into a labeled context block for the prompt."""
    parts = []
    for c in chunks:
        source = c.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{c.page_content}")
    return "\n\n---\n\n".join(parts)


def run_rag(query: str) -> None:
    print(f"Query: {query}\n")

    print("== Retrieving context ==")
    chunks = similarity_search(query)
    for c in chunks:
        print(f"  - {c.metadata['chunk_id']}")

    if not chunks:
        print("\nNo context retrieved — cannot answer.")
        return

    context = build_context(chunks)

    print("\n== Generating answer ==")
    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0,
    )

    messages = [
        ("system", SYSTEM_PROMPT.format(context=context)),
        ("human", query),
    ]
    response = llm.invoke(messages)

    print("\n" + "=" * 40)
    print("ANSWER:")
    print(response.content)
    print("=" * 40)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "How much can I expense for internet per month?"
    run_rag(query)