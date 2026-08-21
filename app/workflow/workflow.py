import logging
from langgraph.graph import END, START, StateGraph
from app.agents.evaluator_agent import evaluator_agent
from app.agents.response_agent import response_agent
from app.agents.retriever_agent import retriever_agent
from app.observability.tracing import configure_observability
from app.workflow.state import GraphState

logger = logging.getLogger(__name__)


def build_workflow():
    # Configure observability before workflow execution.
    configure_observability()

    logger.info("[Workflow] Building LangGraph workflow")

    workflow = StateGraph(GraphState)

    # Register nodes.
    workflow.add_node(
        "retriever",
        retriever_agent,
    )

    workflow.add_node(
        "response",
        response_agent,
    )

    workflow.add_node(
        "evaluator",
        evaluator_agent,
    )

    # Define workflow flow.
    workflow.add_edge(
        START,
        "retriever",
    )

    workflow.add_edge(
        "retriever",
        "response",
    )

    workflow.add_edge(
        "response",
        "evaluator",
    )

    workflow.add_edge(
        "evaluator",
        END,
    )

    app = workflow.compile()

    logger.info("[Workflow] Workflow compiled successfully")

    return app

knowledge_assistant_graph = build_workflow()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    print("\n" + "=" * 60)
    print("WORKFLOW TEST")
    print("=" * 60)

    print("\nWorkflow compiled successfully.")

    print("\nFlow:")
    print("START → Retriever → Response → Evaluator → END")

    print("\nNodes:")

    graph = knowledge_assistant_graph.get_graph()

    for node in graph.nodes:
        print(f"  - {node}")

    print("\n" + "=" * 60)