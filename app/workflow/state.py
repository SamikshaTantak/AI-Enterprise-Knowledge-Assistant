from typing import TypedDict

class ChunkResult(TypedDict):
    chunk_id: str
    source: str
    content: str


class RagasScores(TypedDict):
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None


class GraphState(TypedDict):
    # Input
    query: str

    # Set by Retriever Agent
    retrieved_chunks: list[ChunkResult]

    # Set by Response Agent
    answer: str
    sources: list[str]

    # Set by Evaluator Agent
    ragas_scores: RagasScores | None
    evaluation_summary: str | None