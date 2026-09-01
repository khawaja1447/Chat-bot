"""
Multi-document index: FAISS for dense retrieval, BM25 for lexical, both over the
same chunk list, persisted to disk so a restart does not mean re-embedding.
"""

from __future__ import annotations

import json
import pathlib
import re
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from ragbot.config import EMBED_MODEL
from ragbot.ingest import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    """Lowercase word tokens for BM25. Deliberately simple and dependency-free."""
    return _TOKEN.findall(text.lower())


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the sentence-transformers model once per process.

    Cached here rather than with st.cache_resource so this module stays importable
    outside Streamlit — the eval harness and the tests both need that.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


class DocumentStore:
    """A corpus of chunks with a dense and a lexical index over it."""

    def __init__(self, chunks: list, faiss_index: FAISS):
        self.chunks = chunks
        self.faiss = faiss_index
        self._by_id = {c.chunk_id: c for c in chunks}
        self._rebuild_bm25()

    # ── construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_chunks(cls, chunks: list) -> DocumentStore:
        if not chunks:
            raise ValueError("Cannot build a store from zero chunks.")
        index = FAISS.from_texts(
            [c.text for c in chunks],
            embedding=get_embeddings(),
            metadatas=[c.metadata for c in chunks],
        )
        return cls(chunks, index)

    def add_chunks(self, chunks: list) -> None:
        """Add another document's chunks without re-embedding what is already here."""
        new = [c for c in chunks if c.chunk_id not in self._by_id]
        if not new:
            return
        self.faiss.add_texts(
            [c.text for c in new], metadatas=[c.metadata for c in new]
        )
        self.chunks.extend(new)
        self._by_id.update({c.chunk_id: c for c in new})
        self._rebuild_bm25()

    def remove_document(self, doc_name: str) -> None:
        """
        Drop a document. FAISS deletion by metadata is awkward, so the dense index
        is rebuilt from the remaining chunks — acceptable because removal is rare
        and interactive, unlike ingestion.
        """
        remaining = [c for c in self.chunks if c.doc != doc_name]
        if len(remaining) == len(self.chunks):
            return
        if not remaining:
            raise ValueError("Removing that document would empty the index.")
        rebuilt = DocumentStore.from_chunks(remaining)
        self.chunks, self.faiss = rebuilt.chunks, rebuilt.faiss
        self._by_id = rebuilt._by_id
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

    # ── accessors ─────────────────────────────────────────────────────────────

    @property
    def doc_names(self) -> list:
        seen = {}
        for c in self.chunks:
            seen.setdefault(c.doc, 0)
            seen[c.doc] += 1
        return sorted(seen)

    def stats(self) -> dict:
        pages = {(c.doc, c.page) for c in self.chunks}
        return {
            "documents": len(self.doc_names),
            "pages": len(pages),
            "chunks": len(self.chunks),
        }

    def chunk_at(self, i: int) -> Chunk:
        return self.chunks[i]

    def index_of(self, chunk_id: str) -> int:
        return next(i for i, c in enumerate(self.chunks) if c.chunk_id == chunk_id)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path) -> None:
        """
        Persist to disk. The chunk list is written as JSON and BM25 is rebuilt on
        load — it is fast to build and pickling a third-party index is a liability
        across upgrades.
        """
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.faiss.save_local(str(path / "faiss"))
        (path / "chunks.json").write_text(
            json.dumps(
                [
                    {
                        "text": c.text,
                        "doc": c.doc,
                        "page": c.page,
                        "chunk_id": c.chunk_id,
                    }
                    for c in self.chunks
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path) -> DocumentStore:
        path = pathlib.Path(path)
        payload = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**row) for row in payload]
        index = FAISS.load_local(
            str(path / "faiss"),
            get_embeddings(),
            # Written by this application, on this machine, in save() above.
            allow_dangerous_deserialization=True,
        )
        return cls(chunks, index)

    @staticmethod
    def exists(path) -> bool:
        path = pathlib.Path(path)
        return (path / "chunks.json").is_file() and (path / "faiss").is_dir()
