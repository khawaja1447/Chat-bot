"""Fusion arithmetic, candidate generation, and the retrieve() pipeline."""

from __future__ import annotations

import pytest

from ragbot.config import RagConfig
from ragbot.ingest import Chunk
from ragbot.retrieval import Hit, rerank, retrieve, rrf_fuse, sparse_candidates
from ragbot.store import DocumentStore

NO_RERANK = RagConfig(use_reranker=False, final_k=3, dense_k=10, sparse_k=10)


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def test_rrf_rewards_appearing_in_both_legs():
    """
    The property that makes fusion worth doing: a passage both retrievers found
    outranks one that only a single retriever found, even at a better rank.

    Note this is specifically about *presence in both lists*, not about average
    rank — 1/(k+1) + 1/(k+3) always exceeds 2/(k+2) by convexity, so ranks 1-and-3
    beat 2-and-2. Agreement wins because it adds a term, not because it averages.
    """
    fused = dict(rrf_fuse([["solo", "shared"], ["shared"]], rrf_k=60))
    assert fused["solo"] == pytest.approx(1 / 61)
    assert fused["shared"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["shared"] > fused["solo"]


def test_rrf_scores_are_the_sum_of_reciprocal_ranks():
    fused = dict(rrf_fuse([["a", "b", "c"], ["c", "b", "a"]], rrf_k=60))
    assert fused["b"] == pytest.approx(2 / 62)
    assert fused["a"] == fused["c"] == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_uses_one_based_ranks():
    fused = dict(rrf_fuse([["only"]], rrf_k=60))
    assert fused["only"] == pytest.approx(1 / 61)


def test_rrf_orders_by_descending_score():
    scores = [score for _, score in rrf_fuse([["a", "b", "c"], ["a", "c"]])]
    assert scores == sorted(scores, reverse=True)


def test_rrf_damping_flattens_deep_ranks():
    """A large rrf_k makes rank 1 and rank 2 nearly equivalent; a small one does not."""
    tight = dict(rrf_fuse([["a", "b"]], rrf_k=1))
    loose = dict(rrf_fuse([["a", "b"]], rrf_k=1000))
    assert tight["a"] / tight["b"] > loose["a"] / loose["b"]


def test_rrf_on_empty_input():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


# ── BM25 leg ──────────────────────────────────────────────────────────────────

def test_sparse_finds_an_exact_rare_token(fake_embeddings, sample_chunks):
    store = DocumentStore.from_chunks(sample_chunks)
    ids = sparse_candidates(store, "p99 latency", k=3)
    assert ids[0] == "runbook.pdf::p1::c0"


def test_sparse_returns_nothing_for_an_unmatchable_query(fake_embeddings, sample_chunks):
    store = DocumentStore.from_chunks(sample_chunks)
    assert sparse_candidates(store, "zzzz qqqq", k=5) == []
    assert sparse_candidates(store, "!!! ???", k=5) == []


def test_sparse_respects_the_document_filter(fake_embeddings, sample_chunks):
    store = DocumentStore.from_chunks(sample_chunks)
    ids = sparse_candidates(store, "paid leave roaster", k=5, docs=["handbook.pdf"])
    assert ids and all(i.startswith("handbook.pdf") for i in ids)


# ── retrieve() ────────────────────────────────────────────────────────────────

def test_retrieve_honours_final_k(fake_embeddings, sample_chunks):
    store = DocumentStore.from_chunks(sample_chunks)
    hits = retrieve(store, "revenue", NO_RERANK.variant(final_k=2))
    assert len(hits) == 2


def test_retrieve_records_which_leg_found_each_hit(fake_embeddings, sample_chunks):
    store = DocumentStore.from_chunks(sample_chunks)
    hits = retrieve(store, "p99 latency target", NO_RERANK)
    found = next(h for h in hits if h.chunk.doc == "runbook.pdf")
    assert found.sparse_rank is not None, "BM25 should have surfaced this one"
    assert found.citation == "runbook.pdf p.1"


def test_dense_only_mode_leaves_sparse_ranks_unset(fake_embeddings, sample_chunks):
    store = DocumentStore.from_chunks(sample_chunks)
    hits = retrieve(store, "p99 latency", NO_RERANK.variant(use_hybrid=False))
    assert all(h.sparse_rank is None for h in hits)
    assert all(h.dense_rank is not None for h in hits)


def test_document_filter_excludes_everything_else(fake_embeddings, sample_chunks):
    store = DocumentStore.from_chunks(sample_chunks)
    hits = retrieve(store, "leave", NO_RERANK, docs=["handbook.pdf"])
    assert hits
    assert {h.chunk.doc for h in hits} == {"handbook.pdf"}


def test_hybrid_recovers_a_lexical_match_dense_alone_would_rank_lower(
    fake_embeddings, sample_chunks
):
    """
    The fake embedder hashes words, so a query sharing no tokens with the target
    is effectively invisible to the dense leg. BM25 is what brings it back.
    """
    store = DocumentStore.from_chunks(sample_chunks)
    hybrid = retrieve(store, "p99", NO_RERANK)
    assert any(h.chunk.doc == "runbook.pdf" for h in hybrid)


# ── reranking ─────────────────────────────────────────────────────────────────

def test_rerank_on_empty_input_short_circuits():
    assert rerank("anything", [], top_n=5) == []


def test_rerank_sorts_by_score_and_truncates(monkeypatch):
    chunks = [Chunk(f"text {i}", "d.pdf", i, f"d.pdf::p{i}::c0") for i in range(4)]
    hits = [Hit(chunk=c, score=0.0) for c in chunks]

    # Reverse order: the last candidate is the most relevant.
    monkeypatch.setattr(
        "ragbot.retrieval.get_reranker",
        lambda: type("M", (), {"predict": staticmethod(lambda pairs: [0.0, 1.0, 2.0, 3.0])})(),
    )
    result = rerank("q", hits, top_n=2)

    assert [h.chunk.page for h in result] == [3, 2]
    assert result[0].rerank_score == 3.0


@pytest.mark.slow
def test_real_cross_encoder_separates_relevant_from_irrelevant():
    """Guards the assumption the whole rerank stage rests on. Downloads a model."""
    from ragbot.retrieval import get_reranker

    model = get_reranker()
    good, bad = model.predict(
        [
            ("what was total revenue", "total revenue for 2024 was 18.4 million dollars"),
            ("what was total revenue", "hearing protection is mandatory near a roaster"),
        ]
    )
    assert good > bad
