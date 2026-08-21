import contextvars
import logging
import math
from datasets import Dataset
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness
from ragas.run_config import RunConfig

from app.config.settings import settings
from app.rag.embeddings import get_embedding_model
from app.workflow.state import RagasScores


logger = logging.getLogger(__name__)


def _empty_scores() -> RagasScores:
    """
    Return the default empty RAGAS score structure.
    """

    return {
        "faithfulness": None,
        "answer_relevancy": None,
        "context_precision": None,
        "context_recall": None,
    }


def _safe_score(value: object) -> float | None:

    try:
        if value is None:
            return None

        score = float(value)

        if math.isnan(score) or math.isinf(score):
            return None

        return round(score, 3)

    except (TypeError, ValueError):
        return None


def _interpret(
    metric_name: str,
    score: float | None,
) -> str:
    """
    Build a readable interpretation for one metric.
    """

    if score is None:
        return (
            f"{metric_name}: unavailable "
            f"(metric evaluation failed)"
        )

    if score >= 0.8:
        level = "high"
    elif score >= 0.5:
        level = "moderate"
    else:
        level = "low"

    return f"{metric_name}: {score:.2f} ({level})"


def _run_ragas_in_fresh_context(
    dataset: Dataset,
    metric,
    ragas_llm,
    ragas_embeddings,
    run_config: RunConfig,
):
    """
    Run RAGAS in a fresh ContextVar context.

    When evaluate() is called directly from inside a LangGraph node,
    it can inherit the workflow's active callback/tracing context.

    RAGAS 0.3.x may then successfully execute the metric but fail while
    constructing EvaluationResult with:

        IndexError: list index out of range

    inside ragas.callbacks.parse_run_traces().

    A fresh Context prevents LangGraph's execution-local callback state
    from leaking into the RAGAS evaluation.
    """

    def _run():
        try:
            return evaluate(
                dataset=dataset,
                metrics=[metric],
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                run_config=run_config,
                raise_exceptions=False,
            )

        except TypeError as error:
            # Compatibility fallback for RAGAS versions that do not
            # support raise_exceptions.
            if "raise_exceptions" not in str(error):
                raise

            return evaluate(
                dataset=dataset,
                metrics=[metric],
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                run_config=run_config,
            )

    fresh_context = contextvars.Context()

    return fresh_context.run(_run)

def _evaluate_single_metric(
    dataset: Dataset,
    metric,
    ragas_llm,
    ragas_embeddings,
    run_config: RunConfig,
) -> float | None:
    """
    Evaluate one RAGAS metric independently.

    Each metric is isolated so:
    1. A failure in one metric does not affect another.
    2. RAGAS runs outside the LangGraph execution context.
    """

    metric_name = metric.name

    logger.info(
        "[RAGAS] Evaluating metric: %s",
        metric_name,
    )

    try:
        result = _run_ragas_in_fresh_context(
            dataset=dataset,
            metric=metric,
            ragas_llm=ragas_llm,
            ragas_embeddings=ragas_embeddings,
            run_config=run_config,
        )

        row = result.to_pandas().iloc[0]

        score = _safe_score(
            row.get(metric_name)
        )

        if score is None:
            logger.warning(
                "[RAGAS] Metric '%s' returned no valid score",
                metric_name,
            )
        else:
            logger.info(
                "[RAGAS] %s score: %.3f",
                metric_name,
                score,
            )

        return score

    except Exception:
        logger.exception(
            "[RAGAS] Metric '%s' failed",
            metric_name,
        )

        return None


def evaluate_response(
    query: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
) -> tuple[RagasScores, str]:
   
    scores = _empty_scores()

    if not query.strip():
        summary = (
            "No query was provided, "
            "so RAGAS metrics could not be computed."
        )

        return scores, summary

    if not contexts:
        summary = (
            "No context was retrieved for this query, "
            "so RAGAS metrics could not be computed."
        )

        return scores, summary

    if not answer.strip():
        summary = (
            "No answer was generated, "
            "so RAGAS metrics could not be computed."
        )

        return scores, summary

    try:
        logger.info(
            "[RAGAS] Initializing evaluation components"
        )

        llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0,
            max_tokens=4096,
            max_retries=3,
        )

        ragas_llm = LangchainLLMWrapper(
            llm,
            bypass_temperature=True,
            bypass_n=True,
        )

        ragas_embeddings = LangchainEmbeddingsWrapper(
            get_embedding_model()
        )

        # Keep evaluation context reasonably sized.
        max_contexts = 3
        max_context_chars = 6000

        evaluation_contexts = [
            context[:max_context_chars]
            for context in contexts[:max_contexts]
            if context.strip()
        ]

        if not evaluation_contexts:
            return (
                scores,
                (
                    "No valid retrieved context was available "
                    "for RAGAS evaluation."
                ),
            )

        dataset = Dataset.from_dict(
            {
                "question": [query],
                "answer": [answer],
                "contexts": [evaluation_contexts],
            }
        )

        # Groq-compatible Answer Relevancy configuration.
        answer_relevancy.strictness = 1

        # Sequential evaluation helps avoid rate limits.
        run_config = RunConfig(
            timeout=180,
            max_retries=5,
            max_wait=60,
            max_workers=1,
        )

    except Exception:
        logger.exception(
            "[RAGAS] Failed to initialize evaluation"
        )

        return (
            scores,
            "RAGAS evaluation failed to initialize.",
        )

    logger.info(
        "[RAGAS] Starting independent metric evaluation"
    )

    scores["faithfulness"] = _evaluate_single_metric(
        dataset=dataset,
        metric=faithfulness,
        ragas_llm=ragas_llm,
        ragas_embeddings=ragas_embeddings,
        run_config=run_config,
    )

    scores["answer_relevancy"] = _evaluate_single_metric(
        dataset=dataset,
        metric=answer_relevancy,
        ragas_llm=ragas_llm,
        ragas_embeddings=ragas_embeddings,
        run_config=run_config,
    )

    lines = [
        _interpret(
            "Faithfulness",
            scores["faithfulness"],
        ),
        _interpret(
            "Answer Relevancy",
            scores["answer_relevancy"],
        ),
    ]

    summary = " | ".join(lines)

    logger.info(
        "[RAGAS] Evaluation completed: %s",
        scores,
    )

    return scores, summary


# Backward compatibility with evaluator_agent.py.
evaluate_rag_response = evaluate_response


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    test_scores, test_summary = evaluate_response(
        query=(
            "How much can I expense for internet per month?"
        ),
        answer=(
            "Internet costs are not covered under the company's "
            "expense reimbursement policy."
        ),
        contexts=[
            (
                "3. Non-Reimbursable Expenses\n"
                "The company will not reimburse expenses for "
                "personal internet, home office furniture beyond "
                "the stipend, or commuting costs."
            ),
        ],
    )

    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULT")
    print("=" * 60)

    print("\nScores:")

    for metric, score in test_scores.items():
        print(f"{metric}: {score}")

    print("\nSummary:")
    print(test_summary)

    print("=" * 60)