"""Tunable knobs, in one place, so the eval harness can sweep them."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Groq retires model ids over time and availability differs by account, so this
# is overridable without a code change. `python scripts/list_models.py` prints
# what a given key can actually use.
LLM_MODEL = os.getenv("RAGBOT_LLM_MODEL", "openai/gpt-oss-120b")

MAX_PAGES_PER_DOC = 300
MAX_DOCS = 20


@dataclass(frozen=True)
class RagConfig:
    """
    One retrieval configuration.

    The defaults are the ones the eval harness selected; see docs/EVALUATION.md.
    Chunks are measured in characters, not tokens.
    """

    chunk_size: int = 900
    chunk_overlap: int = 200

    # Candidate generation. Both legs run when hybrid is on, then fuse.
    dense_k: int = 20
    sparse_k: int = 20
    use_hybrid: bool = True

    # Reciprocal Rank Fusion damping. 60 is the value from the original paper;
    # it flattens the contribution of deep ranks so one leg cannot dominate.
    rrf_k: int = 60

    # Reranking narrows the fused candidates down to what the LLM actually sees.
    # The sweep showed no quality gain past ~5 candidates on the sample corpus,
    # only latency; 10 leaves the reranker headroom on a larger one.
    use_reranker: bool = True
    rerank_candidates: int = 10

    # Chunks passed to the LLM.
    final_k: int = 5

    # Rewrite a follow-up into a standalone query before retrieving.
    use_query_rewrite: bool = True
    history_turns: int = 3

    def variant(self, **kwargs) -> RagConfig:
        """A copy with some knobs changed — used to build eval arms."""
        return replace(self, **kwargs)

    @property
    def label(self) -> str:
        if not self.use_hybrid and not self.use_reranker:
            return "dense-only"
        if self.use_hybrid and not self.use_reranker:
            return "hybrid"
        if not self.use_hybrid and self.use_reranker:
            return "dense+rerank"
        return "hybrid+rerank"


DEFAULT = RagConfig()
