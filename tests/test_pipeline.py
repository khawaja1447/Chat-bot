"""History pairing, follow-up rewriting, prompt assembly, streaming."""

from __future__ import annotations

import pytest

from ragbot import pipeline
from ragbot.config import RagConfig
from ragbot.ingest import Chunk
from ragbot.pipeline import (
    answer,
    answer_stream,
    build_context,
    format_history,
    pair_history,
    rewrite_question,
)
from ragbot.retrieval import Hit
from ragbot.store import DocumentStore

CONFIG = RagConfig(use_reranker=False, final_k=2, dense_k=5, sparse_k=5)


# ── history pairing ───────────────────────────────────────────────────────────

def test_pairs_complete_turns():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    assert pair_history(messages) == [("q1", "a1"), ("q2", "a2")]


def test_skips_an_orphaned_question_instead_of_mispairing():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "never answered"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    # Naively zipping alternate elements would pair "never answered" with "a2".
    assert pair_history(messages) == [("q1", "a1"), ("q2", "a2")]


def test_ignores_a_trailing_unanswered_question():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "just asked"},
    ]
    assert pair_history(messages) == [("q1", "a1")]


def test_pair_history_on_empty_log():
    assert pair_history([]) == []


def test_format_history_keeps_only_the_last_n_turns():
    history = [(f"q{i}", f"a{i}") for i in range(10)]
    text = format_history(history, turns=2)
    assert "q9" in text and "q8" in text
    assert "q7" not in text


# ── follow-up rewriting ───────────────────────────────────────────────────────

def test_no_history_means_no_llm_call(monkeypatch):
    def explode(*a, **k):
        raise AssertionError("should not call the LLM with no history")

    monkeypatch.setattr(pipeline.llm, "complete", explode)
    assert rewrite_question("What is revenue?", [], "key") == "What is revenue?"


def test_rewrite_can_be_disabled(monkeypatch):
    monkeypatch.setattr(pipeline.llm, "complete", lambda *a, **k: "REWRITTEN")
    config = RagConfig(use_query_rewrite=False)
    out = rewrite_question("and the second?", [("q", "a")], "key", config)
    assert out == "and the second?"


def test_rewrite_expands_a_follow_up(monkeypatch):
    monkeypatch.setattr(
        pipeline.llm, "complete", lambda *a, **k: "What is the Daily Blend line?"
    )
    out = rewrite_question("and the second one?", [("List the lines", "A, B, C")], "key")
    assert out == "What is the Daily Blend line?"


def test_rewrite_falls_back_when_the_llm_fails(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(pipeline.llm, "complete", boom)
    assert rewrite_question("and the second?", [("q", "a")], "key") == "and the second?"


def test_rewrite_rejects_a_runaway_response(monkeypatch):
    """A rewrite that turns into prose means the model answered instead of rewriting."""
    monkeypatch.setattr(pipeline.llm, "complete", lambda *a, **k: "x" * 500)
    assert rewrite_question("and the second?", [("q", "a")], "key") == "and the second?"


def test_rewrite_rejects_an_empty_response(monkeypatch):
    monkeypatch.setattr(pipeline.llm, "complete", lambda *a, **k: "   ")
    assert rewrite_question("and the second?", [("q", "a")], "key") == "and the second?"


# ── prompt assembly ───────────────────────────────────────────────────────────

def test_context_labels_every_excerpt_with_document_and_page():
    hits = [
        Hit(Chunk("revenue text", "annual.pdf", 3, "a::p3::c0"), 1.0),
        Hit(Chunk("leave text", "handbook.pdf", 2, "h::p2::c0"), 0.9),
    ]
    context = build_context(hits)
    assert "[annual.pdf p.3]" in context
    assert "[handbook.pdf p.2]" in context


# ── answer() ──────────────────────────────────────────────────────────────────

def test_answer_retrieves_with_the_rewritten_query(fake_embeddings, sample_chunks, monkeypatch):
    store = DocumentStore.from_chunks(sample_chunks)
    seen = {}
    calls = []

    def fake_complete(prompt, api_key, **kwargs):
        calls.append(prompt)
        return "22 days." if len(calls) > 1 else "How much paid leave do employees get?"

    monkeypatch.setattr(pipeline.llm, "complete", fake_complete)

    real_retrieve = pipeline.retrieve

    def spy(store_, query, config, docs=None):
        seen["query"] = query
        return real_retrieve(store_, query, config, docs)

    monkeypatch.setattr(pipeline, "retrieve", spy)

    result = answer(store, "and how many?", [("Tell me about leave", "Sure")], "key", CONFIG)

    assert seen["query"] == "How much paid leave do employees get?"
    assert result.search_query == "How much paid leave do employees get?"
    assert result.text == "22 days."
    assert "retrieve_ms" in result.timings


def test_answer_refuses_when_retrieval_comes_back_empty(fake_embeddings, sample_chunks, monkeypatch):
    store = DocumentStore.from_chunks(sample_chunks)
    monkeypatch.setattr(pipeline, "retrieve", lambda *a, **k: [])

    def explode(*a, **k):
        raise AssertionError("must not call the LLM with no context")

    monkeypatch.setattr(pipeline.llm, "complete", explode)

    result = answer(store, "anything?", [], "key", CONFIG)
    assert result.text == "I couldn't find that in the documents."
    assert result.hits == []


# ── streaming ─────────────────────────────────────────────────────────────────

def test_stream_exposes_hits_before_any_token(fake_embeddings, sample_chunks, monkeypatch):
    store = DocumentStore.from_chunks(sample_chunks)
    monkeypatch.setattr(pipeline.llm, "stream", lambda *a, **k: iter(["22 ", "days."]))

    streaming = answer_stream(store, "how much leave?", [], "key", CONFIG)

    assert streaming.hits, "sources must be available before generation starts"
    assert "".join(streaming.tokens()) == "22 days."
    assert "generate_ms" in streaming.timings


def test_stream_refuses_when_retrieval_comes_back_empty(fake_embeddings, sample_chunks, monkeypatch):
    store = DocumentStore.from_chunks(sample_chunks)
    monkeypatch.setattr(pipeline, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(
        pipeline.llm, "stream", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no"))
    )

    streaming = answer_stream(store, "anything?", [], "key", CONFIG)
    assert "".join(streaming.tokens()) == "I couldn't find that in the documents."


def test_missing_api_key_is_reported_clearly():
    from ragbot.llm import MissingAPIKey, complete

    with pytest.raises(MissingAPIKey, match="GROQ_API_KEY"):
        complete("prompt", "")
