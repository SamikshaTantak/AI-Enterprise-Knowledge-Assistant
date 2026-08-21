"""
End-to-end test for the Enterprise Knowledge Assistant workflow.

Flow:
START
  ↓
Retriever Agent
  ↓
Response Agent
  ↓
Evaluator Agent
  ↓
END
"""

import logging

from app.workflow.workflow import knowledge_assistant_graph


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    query = "How much can I expense for internet per month?"

    print("\n" + "=" * 70)
    print("ENTERPRISE KNOWLEDGE ASSISTANT — END-TO-END WORKFLOW TEST")
    print("=" * 70)

    print(f"\nQUERY:\n{query}")

    print("\n" + "-" * 70)
    print("Running workflow...")
    print("-" * 70)

    initial_state = {
        "query": query,
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "ragas_scores": None,
        "evaluation_summary": None,
    }

    try:
        result = knowledge_assistant_graph.invoke(
            initial_state,
            config={
                "run_name": "enterprise_knowledge_workflow_test",
                "tags": [
                    "enterprise-rag",
                    "workflow-test",
                    "langgraph",
                ],
                "metadata": {
                    "environment": "development",
                    "evaluation_enabled": True,
                },
            },
        )

        print("\n" + "=" * 70)
        print("FINAL RESULT")
        print("=" * 70)

        print("\nANSWER:")
        print(result.get("answer", "No answer generated."))

        print("\nSOURCES:")
        sources = result.get("sources", [])

        if sources:
            for source in sources:
                print(f"  - {source}")
        else:
            print("  No sources found.")

        print("\nRAGAS SCORES:")
        ragas_scores = result.get("ragas_scores")

        if ragas_scores:
            for metric, score in ragas_scores.items():
                print(f"  {metric}: {score}")
        else:
            print("  No evaluation scores available.")

        print("\nEVALUATION SUMMARY:")
        print(
            result.get(
                "evaluation_summary",
                "No evaluation summary available.",
            )
        )

        print("\n" + "=" * 70)
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as e:
        logging.exception(
            "[Workflow Test] Workflow execution failed: %s",
            e,
        )


if __name__ == "__main__":
    main()