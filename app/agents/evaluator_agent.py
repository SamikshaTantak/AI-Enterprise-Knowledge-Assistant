import logging

from app.evaluation.ragas_eval import evaluate_rag_response
from app.workflow.state import GraphState


logger = logging.getLogger(__name__)


def evaluator_agent(state: GraphState) -> dict:
    """
    LangGraph node function.

    Evaluates the Response Agent's generated answer against the
    retrieved context using RAGAS.
    """

    query = state["query"]
    answer = state.get("answer", "")
    chunks = state.get("retrieved_chunks", [])

    logger.info(
        "[Evaluator Agent] Starting answer evaluation"
    )

    retrieved_contexts = [
        chunk.get("content", "")
        for chunk in chunks
        if chunk.get("content", "").strip()
    ]
    
    if not retrieved_contexts:
        logger.warning(
            "[Evaluator Agent] No retrieved contexts available"
        )

        scores = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        }

        return {
            "ragas_scores": scores,
            "evaluation_summary": (
                "Evaluation skipped because no retrieved context "
                "was available."
            ),
        }


    if not answer.strip():
        logger.warning(
            "[Evaluator Agent] No answer available for evaluation"
        )

        scores = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        }

        return {
            "ragas_scores": scores,
            "evaluation_summary": (
                "Evaluation skipped because no answer was generated."
            ),
        }
    
    try:

        scores, summary = evaluate_rag_response(
            query=query,
            answer=answer,
            contexts=retrieved_contexts,
        )

        logger.info(
            "[Evaluator Agent] Evaluation completed successfully"
        )

        return {
            "ragas_scores": scores,
            "evaluation_summary": summary,
        }

    except Exception as e:
        logger.exception(
            "[Evaluator Agent] Evaluation failed: %s",
            e,
        )

        scores = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        }

        return {
            "ragas_scores": scores,
            "evaluation_summary": (
                f"Evaluation failed: {str(e)}"
            ),
        }


if __name__ == "__main__":
    """
    Quick standalone test.

    Run:
        uv run python -m app.agents.evaluator_agent
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    test_state: GraphState = {
        "query": (
            "How much can I expense for internet per month?"
        ),
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
        "answer": (
            "Home internet expenses are not reimbursable according to "
            "the expense reimbursement policy."
        ),
        "sources": [
            "expense_reimbursement_policy.txt"
        ],
        "ragas_scores": None,
        "evaluation_summary": None,
    }

    result = evaluator_agent(test_state)

    print("\n" + "=" * 55)
    print("EVALUATION RESULT")
    print("=" * 55)

    print("\nRAGAS SCORES:")

    for metric, score in result["ragas_scores"].items():
        print(f"{metric}: {score}")

    print("\nSUMMARY:")
    print(result["evaluation_summary"])

    print("=" * 55)