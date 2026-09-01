"""Index construction, mutation and on-disk round-trip."""

from __future__ import annotations

import pytest

from ragbot.ingest import Chunk
from ragbot.store import DocumentStore, tokenize


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("The p99 latency, 400ms!") == ["the", "p99", "latency", "400ms"]


def test_tokenize_on_empty_string():
    assert tokenize("   ") == []


def test_from_chunks_rejects_an_empty_corpus(fake_embeddings):
    with pytest.raises(ValueError, match="zero chunks"):
        DocumentStore.from_chunks([])


def test_stats_counts_documents_pages_and_chunks(fake_embeddings, sample_chunks):
    store = DocumentStore.from_chunks(sample_chunks)
    assert store.stats() == {"documents": 3, "pages": 4, "chunks": 4}
    assert store.doc_names == ["annual.pdf", "handbook.pdf", "runbook.pdf"]


def test_add_chunks_extends_both_indexes(fake_embeddings, sample_chunks):
    store = DocumentStore.from_chunks(sample_chunks)
    before = len(store.bm25.doc_freqs)

    store.add_chunks([Chunk("cold brew keeps 21 days", "catalogue.pdf", 2, "catalogue.pdf::p2::c0")])

    assert store.stats()["chunks"] == 5
    assert len(store.bm25.doc_freqs) == before + 1, "BM25 must be rebuilt, not left stale"
    assert "catalogue.pdf" in store.doc_names


def test_add_chunks_is_idempotent(fake_embeddings, sample_chunks):
    store = DocumentStore.from_chunks(sample_chunks)
    store.add_chunks(sample_chunks)          # same chunk_ids
    assert store.stats()["chunks"] == 4


def test_remove_document_drops_only_that_document(fake_embeddings, sample_chunks):
    store = DocumentStore.from_chunks(sample_chunks)
    store.remove_document("handbook.pdf")

    assert store.doc_names == ["annual.pdf", "runbook.pdf"]
    assert store.stats()["chunks"] == 2
    assert len(store.bm25.doc_freqs) == 2


def test_remove_unknown_document_is_a_no_op(fake_embeddings, sample_chunks):
    store = DocumentStore.from_chunks(sample_chunks)
    store.remove_document("nope.pdf")
    assert store.stats()["chunks"] == 4


def test_cannot_remove_the_last_document(fake_embeddings):
    only = [Chunk("solo", "one.pdf", 1, "one.pdf::p1::c0")]
    store = DocumentStore.from_chunks(only)
    with pytest.raises(ValueError, match="empty the index"):
        store.remove_document("one.pdf")


def test_save_and_load_round_trip(fake_embeddings, sample_chunks, tmp_path):
    original = DocumentStore.from_chunks(sample_chunks)
    original.save(tmp_path / "idx")

    assert DocumentStore.exists(tmp_path / "idx")
    restored = DocumentStore.load(tmp_path / "idx")

    assert restored.stats() == original.stats()
    assert restored.doc_names == original.doc_names
    assert [c.chunk_id for c in restored.chunks] == [c.chunk_id for c in original.chunks]
    # BM25 is rebuilt on load rather than pickled, so it must actually be there.
    assert restored.bm25.get_scores(tokenize("latency")).any()


def test_exists_is_false_for_a_missing_or_partial_index(tmp_path):
    assert not DocumentStore.exists(tmp_path / "absent")
    (tmp_path / "partial").mkdir()
    (tmp_path / "partial" / "chunks.json").write_text("[]", encoding="utf-8")
    assert not DocumentStore.exists(tmp_path / "partial"), "faiss dir is also required"


def test_round_trip_preserves_non_ascii(fake_embeddings, tmp_path):
    chunks = [Chunk("café — naïve résumé, 日本語", "unicode.pdf", 1, "unicode.pdf::p1::c0")]
    DocumentStore.from_chunks(chunks).save(tmp_path / "idx")
    restored = DocumentStore.load(tmp_path / "idx")
    assert restored.chunks[0].text == "café — naïve résumé, 日本語"
