from langchain_ollama import OllamaEmbeddings
from app.config.settings import settings


def get_embedding_model() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
    )