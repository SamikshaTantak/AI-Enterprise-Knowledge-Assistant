

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---- Groq LLM ----
    groq_api_key: str
    groq_model: str = "llama-3.1-8b-instant"

    # ---- Ollama (local embeddings) ----
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"

    # ---- ChromaDB ----
    chroma_persist_dir: str = "./data/processed/chroma_db"
    chroma_collection_name: str = "enterprise_knowledge"

    # ---- RAG settings ----
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 4

   # ---- LangSmith Observability ----
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "enterprise-knowledge-assistant"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # ---- MCP ----
    mcp_filesystem_root: str = "./data/raw"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — load once per process."""
    return Settings()


settings = get_settings()