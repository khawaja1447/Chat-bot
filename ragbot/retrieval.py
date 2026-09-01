"""
Hybrid retrieval: dense (FAISS) + lexical (BM25), fused with Reciprocal Rank
Fusion, then reordered by a cross-encoder.

Why each stage exists:

  dense    catches paraphrase — "how much holiday do I get" finding "22 days of
           paid annual leave", which shares no content words with the question.
  BM25     catches the exact rare token — a product name, an error code, "p99" —
           which a 384-dimension embedding tends to smear into its neighbours.
  RRF      combines the two on rank rather than score, so no calibration between
           a cosine similarity and a BM25 score is needed.
  reranker a cross-encoder reads the query and the chunk together instead of
           comparing two independently-computed vectors. It is far more accurate
           and far too slow to run over the whole corpus, which is exactly why it
           goes last, over a short candidate list.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import CrossEncoder

from ragbot.config import RERANK_MODEL, RagConfig
from ragbot.ingest import Chunk
from ragbot.store import DocumentStore, tokenize


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int = None
    sparse_rank: int = None
    rerank_score: float = None

    @property
    def citation(self) -> str:
        return self.chunk.citation


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL, max_length=512, device="cpu")


# ── candidate generation ──────────────────────────────────────────────────────

def dense_candidates(store: DocumentStore, query: str, k: int, docs=None) -> list:
    """Return chunk_ids in dense-similarity order."""
    kwargs = {"k": k}
    if docs:
        # fetch_k widens the pool FAISS filters down from, otherwise a filter can
        # come back nearly empty on a large corpus.
        kwargs["filter"] = {"doc": list(docs)} if len(docs) > 1 else {"doc": docs[0]}
        kwargs["fetch_k"] = max(k * 8, 100)
    results = store.faiss.similarity_search(query, **kwargs)
    return [d.metadata["chunk_id"] for d in results]


def sparse_candidates(store: DocumentStore, query: str, k: int, docs=None) -> list:
    """Return chunk_ids in BM25 order."""
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = store.bm25.get_scores(tokens)
    allowed = set(docs) if docs else None

    ranked = sorted(
        (
            (score, i)
            for i, score in enumerate(scores)
            if score > 0 and (allowed is None or store.chunks[i].doc in allowed)
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return [store.chunks[i].chunk_id for _, i in ranked[:k]]


# ── fusion ────────────────────────────────────────────────────────────────────

def rrf_fuse(rankings: list, rrf_k: int = 60) -> list:
    """
    Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (rrf_k + rank(d)).

    Operates on ranks, so a BM25 score of 14.2 and a cosine similarity of 0.83
    never have to be made commensurate. rrf_k damps the tail: with rrf_k=60 the
    gap between rank 1 and rank 2 matters far more than 20 versus 21.

    `rankings` is a list of chunk_id lists, best first.
    Returns [(chunk_id, score)] best first.
    """
    scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


# ── reranking ─────────────────────────────────────────────────────────────────

def rerank(query: str, hits: list, top_n: int) -> list:
    """Reorder hits by cross-encoder relevance and keep the best top_n."""
    if not hits:
        return []
    model = get_reranker()
    scores = model.predict([(query, h.chunk.text) for h in hits])
    for hit, score in zip(hits, scores, strict=True):
        hit.rerank_score = float(score)
    return sorted(hits, key=lambda h: h.rerank_score, reverse=True)[:top_n]


# ── the pipeline ──────────────────────────────────────────────────────────────

def retrieve(
    store: DocumentStore,
    query: str,
    config: RagConfig,
    docs=None,
) -> list:
    """
    Run the configured retrieval pipeline and return the final Hits, best first.

    `docs` optionally restricts the search to a subset of document names.
    """
    dense = dense_candidates(store, query, config.dense_k, docs)

    if config.use_hybrid:
        sparse = sparse_candidates(store, query, config.sparse_k, docs)
        fused = rrf_fuse([dense, sparse], config.rrf_k)
        dense_pos = {cid: i + 1 for i, cid in enumerate(dense)}
        sparse_pos = {cid: i + 1 for i, cid in enumerate(sparse)}
    else:
        fused = [(cid, 1.0 / (config.rrf_k + i)) for i, cid in enumerate(dense, start=1)]
        dense_pos = {cid: i + 1 for i, cid in enumerate(dense)}
        sparse_pos = {}

    by_id = {c.chunk_id: c for c in store.chunks}
    hits = [
        Hit(
            chunk=by_id[cid],
            score=score,
            dense_rank=dense_pos.get(cid),
            sparse_rank=sparse_pos.get(cid),
        )
        for cid, score in fused
        if cid in by_id
    ]

    if config.use_reranker:
        return rerank(query, hits[: config.rerank_candidates], config.final_k)
    return hits[: config.final_k]
