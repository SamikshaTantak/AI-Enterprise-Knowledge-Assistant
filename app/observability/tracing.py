import logging
import os

from app.config.settings import settings


logger = logging.getLogger(__name__)


def configure_observability() -> bool:
    tracing_enabled = settings.langchain_tracing_v2
    api_key = settings.langchain_api_key
    project_name = settings.langchain_project
    endpoint = settings.langchain_endpoint

    if not tracing_enabled:
        logger.info(
            "[Observability] LangSmith tracing is disabled. "
            "Using manual logging."
        )
        return False

    if not api_key:
        logger.warning(
            "[Observability] LANGCHAIN_TRACING_V2 is enabled, "
            "but LANGCHAIN_API_KEY is missing. "
            "Using manual logging."
        )
        return False

    # Configure LangSmith before LangGraph workflow execution.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project_name
    os.environ["LANGCHAIN_ENDPOINT"] = endpoint

    # Modern LangSmith-compatible environment variables.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = project_name
    os.environ["LANGSMITH_ENDPOINT"] = endpoint

    logger.info(
        "[Observability] LangSmith tracing enabled. "
        "Project: %s",
        project_name,
    )

    return True