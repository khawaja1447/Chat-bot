"""
List the Groq models your API key can actually use.

Model availability changes over time and differs by account, so a hard-coded id
eventually 404s. Run this to see what is live for you right now:

    python scripts/list_models.py

Reads GROQ_API_KEY from the environment or .env. The key is never printed.
"""

from __future__ import annotations

import os
import pathlib
import sys

from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ragbot.config import LLM_MODEL  # noqa: E402

load_dotenv()


def main() -> int:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print(
            "No GROQ_API_KEY found.\n"
            "Put it in .env (cp .env.example .env) or export it, then re-run."
        )
        return 1

    try:
        models = Groq(api_key=api_key).models.list().data
    except Exception as exc:
        print(f"Could not reach Groq: {exc}")
        return 1

    chat = sorted(
        (m for m in models if not _is_non_chat(m.id)), key=lambda m: m.id
    )
    other = sorted((m for m in models if _is_non_chat(m.id)), key=lambda m: m.id)

    print(f"{len(models)} models available to this key.\n")
    print("Chat models (usable as RAGBOT_LLM_MODEL):")
    for model in chat:
        marker = "  <-- current default" if model.id == LLM_MODEL else ""
        window = getattr(model, "context_window", None)
        window_text = f"{window:>7,} ctx" if window else " " * 11
        print(f"  {model.id:<45} {window_text}{marker}")

    if other:
        print("\nNon-chat (speech, guard, embedding):")
        for model in other:
            print(f"  {model.id}")

    if LLM_MODEL not in {m.id for m in models}:
        print(
            f"\n!! The configured default '{LLM_MODEL}' is NOT in that list.\n"
            "   Pick one above and set it, e.g. in .env:\n"
            f"       RAGBOT_LLM_MODEL={chat[0].id if chat else '<model-id>'}"
        )
    return 0


def _is_non_chat(model_id: str) -> bool:
    marks = ("whisper", "tts", "guard", "embed", "playai", "prompt-guard")
    return any(mark in model_id.lower() for mark in marks)


if __name__ == "__main__":
    raise SystemExit(main())
