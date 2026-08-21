import importlib.util
import sys
from pathlib import Path

OLD_IMPORT = "from langchain_community.chat_models.vertexai import ChatVertexAI"
NEW_IMPORT = "from langchain_google_vertexai import ChatVertexAI"


def find_ragas_base_file() -> Path | None:
    spec = importlib.util.find_spec("ragas")
    if spec is None or not spec.submodule_search_locations:
        return None
    ragas_root = Path(list(spec.submodule_search_locations)[0])
    target = ragas_root / "llms" / "base.py"
    return target if target.exists() else None


def main() -> None:
    target = find_ragas_base_file()
    if target is None:
        print("[FAIL] Could not locate ragas/llms/base.py — is ragas installed?")
        print("       Run: uv add ragas==0.3.8")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    if NEW_IMPORT in content:
        print(f"[OK] Patch already applied: {target}")
        return

    if OLD_IMPORT not in content:
        print(
            f"[SKIP] Old import string not found in {target}\n"
            "       ragas version may have changed — this patch may no "
            "longer be needed. If ragas still fails to import, check the "
            "current traceback for the actual missing module."
        )
        return

    try:
        importlib.util.find_spec("langchain_google_vertexai")
    except ModuleNotFoundError:
        pass
    if importlib.util.find_spec("langchain_google_vertexai") is None:
        print("[FAIL] langchain_google_vertexai is not installed.")
        print("       Run: uv add langchain-google-vertexai")
        sys.exit(1)

    patched = content.replace(OLD_IMPORT, NEW_IMPORT)
    target.write_text(patched, encoding="utf-8")
    print(f"[OK] Patched {target}")
    print(f"     Old: {OLD_IMPORT}")
    print(f"     New: {NEW_IMPORT}")


if __name__ == "__main__":
    main()