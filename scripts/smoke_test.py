"""
Phase 1 smoke test.

Run this after setting up .env to confirm:
  1. Settings load correctly from .env
  2. Groq API is reachable and responds
  3. Ollama is running locally and the embedding model is pulled
  4. ChromaDB can create a local persistent client

Usage:
    uv run python scripts/smoke_test.py
"""

import sys

from app.config.settings import settings


def check_settings() -> bool:
    print("== [1/4] Checking settings load ==")
    try:
        print(f"  Groq model:        {settings.groq_model}")
        print(f"  Ollama embed model: {settings.ollama_embed_model}")
        print(f"  Chroma dir:        {settings.chroma_persist_dir}")
        if not settings.groq_api_key or "your_groq" in settings.groq_api_key:
            print("  [FAIL] GROQ_API_KEY is missing or still a placeholder in .env")
            return False
        print("  [OK] Settings loaded")
        return True
    except Exception as e:
        print(f"  [FAIL] Could not load settings: {e}")
        return False


def check_groq() -> bool:
    print("\n== [2/4] Checking Groq LLM connectivity ==")
    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0,
        )
        response = llm.invoke("Reply with exactly one word: OK")
        print(f"  Response: {response.content.strip()}")
        print("  [OK] Groq reachable")
        return True
    except Exception as e:
        print(f"  [FAIL] Groq call failed: {e}")
        return False


def check_ollama() -> bool:
    print("\n== [3/4] Checking Ollama embedding model ==")
    try:
        from langchain_ollama import OllamaEmbeddings

        embedder = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embed_model,
        )
        vector = embedder.embed_query("test sentence")
        print(f"  Embedding dimension: {len(vector)}")
        print("  [OK] Ollama reachable and model responds")
        return True
    except Exception as e:
        print(f"  [FAIL] Ollama call failed: {e}")
        print(
            "  -> Is 'ollama' running? Try: ollama list  "
            "(and confirm nomic-embed-text is pulled)"
        )
        return False


def check_chroma() -> bool:
    print("\n== [4/4] Checking ChromaDB local persistence ==")
    try:
        import chromadb

        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        collection = client.get_or_create_collection(settings.chroma_collection_name)
        print(f"  Collection '{collection.name}' ready at {settings.chroma_persist_dir}")
        print("  [OK] ChromaDB working")
        return True
    except Exception as e:
        print(f"  [FAIL] ChromaDB init failed: {e}")
        return False


def main() -> None:
    results = [
        check_settings(),
        check_groq(),
        check_ollama(),
        check_chroma(),
    ]
    print("\n" + "=" * 40)
    if all(results):
        print("ALL CHECKS PASSED — ready for Phase 2")
        sys.exit(0)
    else:
        print("SOME CHECKS FAILED — fix the [FAIL] items above before Phase 2")
        sys.exit(1)


if __name__ == "__main__":
    main()