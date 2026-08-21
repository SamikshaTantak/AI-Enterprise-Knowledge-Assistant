import functools
import logging
import time

logger = logging.getLogger(__name__)


class KnowledgeAssistantError(Exception):
    """Base class for all application-specific errors."""


class LLMServiceError(KnowledgeAssistantError):
    """Raised when the Groq LLM API is unreachable or returns an error."""


class EmbeddingServiceError(KnowledgeAssistantError):
    """Raised when the local Ollama embedding service is unreachable."""


class VectorStoreError(KnowledgeAssistantError):
    """Raised when ChromaDB read/write operations fail."""


class MCPError(KnowledgeAssistantError):
    """Raised when the MCP document server is unreachable, times out, or errors."""


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
):
    """
    Retry a function call with exponential backoff.

    Args:
        max_attempts: Total attempts before giving up (including the first).
        base_delay: Seconds before the first retry; doubles each subsequent
                    attempt (1s, 2s, 4s, ...).
        exceptions: Exception types that trigger a retry. Anything else
                    propagates immediately without retrying.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        break
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"[Retry] {func.__name__} failed (attempt "
                        f"{attempt}/{max_attempts}): {e}. Retrying in "
                        f"{delay:.1f}s..."
                    )
                    time.sleep(delay)
            raise last_exception

        return wrapper

    return decorator


def user_facing_message(error: Exception) -> str:

    if isinstance(error, LLMServiceError):
        return (
            "The AI language service is temporarily unavailable. "
            "This is usually a rate limit or connectivity issue — "
            "please try again in a moment."
        )
    if isinstance(error, EmbeddingServiceError):
        return (
            "The local embedding service (Ollama) isn't responding. "
            "Please confirm Ollama is running on this machine."
        )
    if isinstance(error, VectorStoreError):
        return (
            "The knowledge base (vector store) couldn't be reached. "
            "Try re-running ingestion, or restart the application."
        )
    if isinstance(error, MCPError):
        return (
            "The document server (MCP) is unavailable or timed out. "
            "Live document lookups are temporarily disabled, but "
            "cached vector search results may still work."
        )
    return (
        "Something unexpected went wrong while processing your request. "
        "Please try again — if this keeps happening, check the application logs."
    )