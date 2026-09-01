"""Shared fixtures.

The fast suite never loads a real model: embeddings are faked deterministically so
tests stay offline and quick. Tests that genuinely need the sentence-transformers
models are marked `slow` and select in explicitly.
"""

from __future__ import annotations

import hashlib
import io

import pymupdf
import pytest
from langchain_core.embeddings import Embeddings

from ragbot.ingest import Chunk


class FakeEmbeddings(Embeddings):
    """
    Deterministic bag-of-words hashing embedder.

    Not semantically meaningful, but stable and fast: two texts sharing words land
    near each other, which is enough to exercise the plumbing.
    """

    dim = 64

    def _vector(self, text: str) -> list:
        vec = [0.0] * self.dim
        for word in text.lower().split():
            slot = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            vec[slot] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list) -> list:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list:
        return self._vector(text)


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Point the store at the fake embedder for the duration of a test."""
    embedder = FakeEmbeddings()
    monkeypatch.setattr("ragbot.store.get_embeddings", lambda: embedder)
    return embedder


def make_pdf(pages) -> io.BytesIO:
    """Build an in-memory PDF. A page given as None is left blank (image-only)."""
    doc = pymupdf.open()
    for body in pages:
        page = doc.new_page()
        if body:
            page.insert_textbox(
                pymupdf.Rect(50, 50, 550, 780), body, fontsize=11, fontname="helv"
            )
    buf = io.BytesIO(doc.tobytes())
    doc.close()
    return buf


@pytest.fixture
def pdf_factory():
    return make_pdf


@pytest.fixture
def sample_chunks():
    return [
        Chunk("annual revenue was 18.4 million dollars", "annual.pdf", 3, "annual.pdf::p3::c0"),
        Chunk("employees accrue 22 days of paid leave", "handbook.pdf", 2, "handbook.pdf::p2::c0"),
        Chunk("the p99 latency target is 400 milliseconds", "runbook.pdf", 1, "runbook.pdf::p1::c0"),
        Chunk("hearing protection is mandatory near a roaster", "handbook.pdf", 4, "handbook.pdf::p4::c0"),
    ]
