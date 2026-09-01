"""
PDF Q&A over a document library — Streamlit UI.

The interesting parts of this project are in ragbot/; this file is presentation
and session state only.
"""

from __future__ import annotations

import os
import pathlib

import streamlit as st
from dotenv import load_dotenv

from ragbot.config import DEFAULT, LLM_MODEL, MAX_DOCS, RagConfig
from ragbot.ingest import IngestError, chunk_document, extract_document
from ragbot.observability import configure as configure_logging
from ragbot.pipeline import answer_stream, pair_history
from ragbot.store import DocumentStore

load_dotenv()
configure_logging()

INDEX_PATH = pathlib.Path(os.getenv("RAGBOT_INDEX_PATH", ".index"))

st.set_page_config(
    page_title="PDF Q&A — Hybrid RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.1rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-header { color: #6b7280; font-size: 0.95rem; margin-bottom: 1.2rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state ─────────────────────────────────────────────────────────────

def init_state() -> None:
    defaults = {
        "store": None,
        "chat_history": [],
        "doc_filter": [],
        "loaded_from_disk": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def secret(name: str) -> str:
    """st.secrets raises outright when no secrets.toml exists — the normal local case."""
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


def restore_index() -> None:
    """Load a previously built index once per session, so a restart is cheap."""
    if st.session_state.store is not None or st.session_state.loaded_from_disk:
        return
    st.session_state.loaded_from_disk = True
    if DocumentStore.exists(INDEX_PATH):
        try:
            st.session_state.store = DocumentStore.load(INDEX_PATH)
        except Exception as exc:            # a stale index must not brick the app
            st.session_state.store = None
            st.warning(f"Could not load the saved index ({exc}). Upload documents to rebuild.")


restore_index()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    typed_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Free key at https://console.groq.com/keys. Leave blank to use "
             "GROQ_API_KEY from .env or Streamlit secrets.",
    )
    api_key = typed_key or secret("GROQ_API_KEY") or os.getenv("GROQ_API_KEY", "")
    if api_key and not typed_key:
        st.caption("🔑 Using GROQ_API_KEY from the environment.")

    with st.expander("Retrieval settings"):
        use_hybrid = st.toggle(
            "Hybrid search (BM25 + dense)", value=DEFAULT.use_hybrid,
            help="Adds lexical matching, which catches exact rare terms that "
                 "embeddings smear together.",
        )
        use_reranker = st.toggle(
            "Cross-encoder reranking", value=DEFAULT.use_reranker,
            help="Reads the query and each candidate together. Much more "
                 "accurate ordering, ~250 ms.",
        )
        final_k = st.slider("Passages sent to the model", 3, 10, DEFAULT.final_k)

    config: RagConfig = DEFAULT.variant(
        use_hybrid=use_hybrid, use_reranker=use_reranker, final_k=final_k
    )

    st.markdown("---")
    st.markdown("## 📚 Library")

    uploads = st.file_uploader(
        "Add PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Up to {MAX_DOCS} documents. Embeddings run locally on CPU.",
    )

    if uploads and st.button("🚀 Index documents", use_container_width=True, type="primary"):
        existing = set(st.session_state.store.doc_names) if st.session_state.store else set()
        added, skipped, failed = [], [], []

        progress = st.progress(0.0, text="Starting…")
        for i, upload in enumerate(uploads, start=1):
            progress.progress(i / len(uploads), text=f"Indexing {upload.name}…")
            if upload.name in existing:
                skipped.append(upload.name)
                continue
            try:
                document = extract_document(upload, upload.name)
                chunks = chunk_document(document, config)
                if st.session_state.store is None:
                    st.session_state.store = DocumentStore.from_chunks(chunks)
                else:
                    st.session_state.store.add_chunks(chunks)
                added.append(f"{upload.name} ({document.page_count}p)")
            except IngestError as exc:
                failed.append(str(exc))
            except Exception as exc:
                failed.append(f"{upload.name}: {exc}")
        progress.empty()

        if added:
            st.session_state.store.save(INDEX_PATH)
            st.session_state.chat_history = []
            st.success("Indexed " + ", ".join(added))
        if skipped:
            st.info("Already indexed: " + ", ".join(skipped))
        for message in failed:
            st.error(message)
        if added:
            st.rerun()

    store: DocumentStore | None = st.session_state.store

    if store is not None:
        stats = store.stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Docs", stats["documents"])
        c2.metric("Pages", stats["pages"])
        c3.metric("Chunks", stats["chunks"])

        st.session_state.doc_filter = st.multiselect(
            "Search only these documents",
            options=store.doc_names,
            default=st.session_state.doc_filter,
            help="Leave empty to search the whole library.",
        )

        with st.expander("Manage documents"):
            for name in store.doc_names:
                row_a, row_b = st.columns([4, 1])
                row_a.caption(name)
                if row_b.button("🗑️", key=f"rm::{name}", help=f"Remove {name}"):
                    try:
                        store.remove_document(name)
                        store.save(INDEX_PATH)
                        st.session_state.chat_history = []
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.caption(
        f"BM25 + FAISS → Reciprocal Rank Fusion → cross-encoder rerank → `{LLM_MODEL}` on Groq. "
        "Embeddings run locally; only the question and retrieved excerpts leave the machine."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">📄 PDF Q&A — Hybrid RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask across a library of PDFs. Every answer cites the '
    "document and page it came from.</div>",
    unsafe_allow_html=True,
)

if store is None:
    st.info("👈 Add one or more PDFs in the sidebar to get started.")
    left, right = st.columns(2)
    with left:
        st.markdown(
            "#### How retrieval works\n"
            "1. **Dense** — FAISS over local MiniLM embeddings finds passages that mean "
            "the same thing, even with no words in common.\n"
            "2. **Lexical** — BM25 catches the exact rare token an embedding blurs: a SKU, "
            "an error code, `p99`.\n"
            "3. **Fusion** — Reciprocal Rank Fusion merges the two on rank, so a BM25 score "
            "and a cosine similarity never have to be made comparable.\n"
            "4. **Rerank** — a cross-encoder reads the question and each candidate together "
            "and reorders them.\n"
            f"5. **Answer** — `{LLM_MODEL}` on Groq writes from the top passages and cites each one."
        )
    with right:
        st.markdown(
            "#### Measured, not asserted\n"
            "`eval/golden_set.yaml` holds 109 questions labelled with the document and page "
            "that actually answers them. `eval/run_eval.py` scores retrieval against it with "
            "no LLM in the loop.\n\n"
            "See `docs/EVALUATION.md` for the current numbers, including the cases that still "
            "fail.\n\n"
            "A sample corpus is in `samples/` — six documents that deliberately overlap, "
            "including two annual reports whose figures differ only by year."
        )
    st.stop()


def render_sources(hits: list, container=None) -> None:
    target = container or st
    if not hits:
        return
    with target.expander(f"📎 {len(hits)} passages used", expanded=False):
        for i, hit in enumerate(hits, start=1):
            bits = [f"**{i}. {hit.chunk.doc} — p. {hit.chunk.page}**"]
            trace = []
            if hit.dense_rank:
                trace.append(f"dense #{hit.dense_rank}")
            if hit.sparse_rank:
                trace.append(f"bm25 #{hit.sparse_rank}")
            if hit.rerank_score is not None:
                trace.append(f"rerank {hit.rerank_score:+.2f}")
            if trace:
                bits.append("  \n`" + "  ·  ".join(trace) + "`")
            st.markdown("".join(bits))
            st.text(hit.chunk.text[:400].strip().replace("\n", " ") + "…")


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("hits"):
            render_sources(message["hits"])
        if message.get("rewritten"):
            st.caption(f"↻ searched for: _{message['rewritten']}_")


def handle_question(question: str) -> bool:
    """Returns True when an answer was produced; False means an error is on screen."""
    history = pair_history(st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving…"):
                streaming = answer_stream(
                    store,
                    question,
                    history,
                    api_key,
                    config,
                    docs=st.session_state.doc_filter or None,
                )
            text = st.write_stream(streaming.tokens())
        except Exception as exc:
            st.session_state.chat_history.pop()
            st.error(f"❌ {exc}")
            # Do not rerun: a rerun would wipe this message before it is read.
            return False

        render_sources(streaming.hits)
        rewritten = (
            streaming.search_query if streaming.search_query != question else None
        )
        if rewritten:
            st.caption(f"↻ searched for: _{rewritten}_")

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": text,
            "hits": streaming.hits,
            "rewritten": rewritten,
        }
    )
    return True


if not st.session_state.chat_history:
    st.markdown("##### 💡 Try asking")
    suggestions = [
        "What was total revenue in 2024?",
        "How many days of paid holiday do I get?",
        "The nightly billing job died halfway. How do I restart it?",
    ]
    for column, suggestion in zip(st.columns(len(suggestions)), suggestions, strict=True):
        if column.button(suggestion, use_container_width=True) and handle_question(suggestion):
            st.rerun()

if prompt := st.chat_input("Ask a question about your documents…"):
    if prompt.strip() and handle_question(prompt.strip()):
        st.rerun()
