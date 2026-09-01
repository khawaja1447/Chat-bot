"""Groq chat completions, blocking and streaming."""

from __future__ import annotations

from groq import Groq

from ragbot.config import LLM_MODEL


class MissingAPIKey(RuntimeError):
    pass


class ModelUnavailable(RuntimeError):
    pass


def _translate(exc: Exception) -> Exception:
    """
    Turn Groq's API errors into something a user can act on.

    A 404 here almost always means the model id has been retired rather than that
    anything is wrong with the request, and the raw error does not say what to do
    about it.
    """
    text = str(exc)
    if "model_not_found" in text or "does not exist" in text:
        return ModelUnavailable(
            f"Groq has no model '{LLM_MODEL}' available to this key — it was most "
            "likely retired.\n\n"
            "Run `python scripts/list_models.py` to see what your key can use, "
            "then set the one you want in .env:\n"
            "    RAGBOT_LLM_MODEL=<model-id>\n"
            "and restart the app."
        )
    return exc


def _client(api_key: str) -> Groq:
    if not api_key:
        raise MissingAPIKey(
            "No Groq API key. Paste one in the sidebar, or set GROQ_API_KEY "
            "in .env / Streamlit secrets."
        )
    return Groq(api_key=api_key)


def complete(prompt: str, api_key: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
    try:
        response = _client(api_key).chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        raise _translate(exc) from exc
    return response.choices[0].message.content or ""


def stream(prompt: str, api_key: str, max_tokens: int = 1024, temperature: float = 0.2):
    """Yield content deltas as they arrive."""
    try:
        completion = _client(api_key).chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
    except Exception as exc:
        raise _translate(exc) from exc
    for chunk in completion:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
