"""Question in, grounded answer out: rewrite -> retrieve -> generate."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ragbot import llm
from ragbot.config import DEFAULT, RagConfig
from ragbot.observability import log_event
from ragbot.retrieval import retrieve
from ragbot.store import DocumentStore

ANSWER_PROMPT = """You answer questions strictly from the document excerpts below.

Rules:
- Use only the excerpts. If they do not contain the answer, say exactly:
  "I couldn't find that in the documents."
- Cite the source after each claim, in the form [document p.N], using the labels
  shown on the excerpts. Cite every excerpt you actually used.
- Do not invent page numbers or documents.
- If the excerpts disagree, say so and cite both.
- Be concise. Do not restate the question.

EXCERPTS:
{context}

CONVERSATION SO FAR:
{history}
QUESTION: {question}

ANSWER:"""

REWRITE_PROMPT = """Rewrite the follow-up question so it can be understood on its own.

Resolve every pronoun and back-reference ("it", "that one", "the second", "there")
into explicit terms taken from the conversation. Keep the user's intent exactly.
If the question already stands alone, repeat it unchanged.
Reply with the rewritten question and nothing else.

CONVERSATION:
{history}
FOLLOW-UP: {question}

STANDALONE QUESTION:"""


@dataclass
class Answer:
    question: str
    search_query: str
    text: str
    hits: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)


def format_history(history: list, turns: int) -> str:
    return "".join(
        f"User: {u}\nAssistant: {a}\n\n" for u, a in history[-turns:]
    )


def build_context(hits: list) -> str:
    return "\n\n---\n\n".join(
        f"[{h.chunk.doc} p.{h.chunk.page}]\n{h.chunk.text}" for h in hits
    )


def pair_history(messages: list) -> list:
    """
    Flat [{role, content}, ...] -> [(user, assistant), ...].

    A message without its counterpart is skipped rather than paired with the
    wrong turn, which is what happens if you simply zip alternate elements.
    """
    pairs = []
    i = 0
    while i < len(messages) - 1:
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            pairs.append((messages[i]["content"], messages[i + 1]["content"]))
            i += 2
        else:
            i += 1
    return pairs


def rewrite_question(
    question: str, history: list, api_key: str, config: RagConfig = DEFAULT
) -> str:
    """
    Turn a follow-up into a standalone query before retrieval.

    Without this, "and the second one?" is embedded literally and retrieves
    whatever happens to sit near it in vector space. Passing history to the
    generation step does not help: by then the wrong chunks are already fetched.
    Falls back to the original question if the rewrite fails.
    """
    if not history or not config.use_query_rewrite:
        return question
    prompt = REWRITE_PROMPT.format(
        history=format_history(history, config.history_turns), question=question
    )
    try:
        rewritten = llm.complete(prompt, api_key, max_tokens=200, temperature=0.0).strip()
    except Exception as exc:
        log_event("rewrite_failed", error=str(exc))
        return question
    # A rewrite that balloons is usually the model ignoring the instruction and
    # answering instead; the raw question is safer than a paragraph of prose.
    if not rewritten or len(rewritten) > 400:
        return question
    return rewritten


def _prepare(store, question, history, api_key, config, docs):
    """Shared front half: rewrite, then retrieve."""
    t0 = time.perf_counter()
    search_query = rewrite_question(question, history, api_key, config)
    t1 = time.perf_counter()
    hits = retrieve(store, search_query, config, docs)
    t2 = time.perf_counter()

    timings = {
        "rewrite_ms": round((t1 - t0) * 1000, 1),
        "retrieve_ms": round((t2 - t1) * 1000, 1),
    }
    log_event(
        "retrieval",
        question=question,
        search_query=search_query,
        rewritten=search_query != question,
        docs_filter=list(docs) if docs else None,
        config=config.label,
        hits=[h.citation for h in hits],
        **timings,
    )
    return search_query, hits, timings


def answer(
    store: DocumentStore,
    question: str,
    history: list,
    api_key: str,
    config: RagConfig = DEFAULT,
    docs=None,
) -> Answer:
    search_query, hits, timings = _prepare(store, question, history, api_key, config, docs)

    if not hits:
        return Answer(question, search_query, "I couldn't find that in the documents.", [], timings)

    prompt = ANSWER_PROMPT.format(
        context=build_context(hits),
        history=format_history(history, config.history_turns),
        question=question,
    )
    t0 = time.perf_counter()
    text = llm.complete(prompt, api_key)
    timings["generate_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    log_event("answer", chars=len(text), **timings)
    return Answer(question, search_query, text, hits, timings)


@dataclass
class StreamingAnswer:
    """Retrieval has already run; `tokens()` streams the generated answer."""

    question: str
    search_query: str
    hits: list
    timings: dict
    _prompt: str
    _api_key: str

    def tokens(self):
        if not self.hits:
            yield "I couldn't find that in the documents."
            return
        t0 = time.perf_counter()
        yield from llm.stream(self._prompt, self._api_key)
        self.timings["generate_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        log_event("answer_streamed", **self.timings)


def answer_stream(
    store: DocumentStore,
    question: str,
    history: list,
    api_key: str,
    config: RagConfig = DEFAULT,
    docs=None,
) -> StreamingAnswer:
    """
    Retrieve eagerly, generate lazily.

    Returning the hits before the first token lets the UI show which passages it
    is about to reason over while the answer is still being written.
    """
    search_query, hits, timings = _prepare(store, question, history, api_key, config, docs)
    prompt = ANSWER_PROMPT.format(
        context=build_context(hits),
        history=format_history(history, config.history_turns),
        question=question,
    )
    return StreamingAnswer(question, search_query, hits, timings, prompt, api_key)
