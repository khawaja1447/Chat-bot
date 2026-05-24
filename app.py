"""
PDF Q&A RAG Chatbot — Streamlit UI
"""

import streamlit as st
from rag_engine import process_pdf, ask

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .chat-message-user {
        background: #1e3a5f;
        border-left: 4px solid #3b82f6;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        color: #e8f0fe !important;
    }
    .chat-message-bot {
        background: #1a3a2a;
        border-left: 4px solid #22c55e;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        color: #d1fae5 !important;
    }
    .source-box {
        background: #2d2a12;
        border: 1px solid #a16207;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        color: #fef08a !important;
        font-size: 0.8rem;
        margin-top: 0.4rem;
    }
    .stat-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "vector_store": None,
        "chat_history": [],          # list of {"role": ..., "content": ..., "sources": ...}
        "pdf_processed": False,
        "pdf_name": "",
        "chunk_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at https://console.groq.com/keys",
    )

    if api_key:
        import os
        os.environ["GROQ_API_KEY"] = api_key

    st.markdown("---")
    st.markdown("## 📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Max recommended: ~50 pages for best performance",
    )

    if uploaded_file and api_key:
        if st.button("🚀 Process PDF", use_container_width=True, type="primary"):
            with st.spinner("Reading & indexing your PDF..."):
                try:
                    vs, n_chunks = process_pdf(uploaded_file)
                    st.session_state.vector_store = vs
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chunk_count = n_chunks
                    st.session_state.chat_history = []
                    st.success(f"✅ Ready! Indexed **{n_chunks}** chunks.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    elif uploaded_file and not api_key:
        st.warning("⚠️ Enter your API key first.")

    if st.session_state.pdf_processed:
        st.markdown("---")
        st.markdown("### 📊 Document Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div class="stat-card">
                <div style="font-size:1.4rem">📄</div>
                <div style="font-weight:600">{st.session_state.pdf_name[:18]}…</div>
                <div style="font-size:0.75rem;color:#6b7280">File</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="stat-card">
                <div style="font-size:1.4rem">🧩</div>
                <div style="font-weight:600">{st.session_state.chunk_count}</div>
                <div style="font-size:0.75rem;color:#6b7280">Chunks</div>
            </div>""", unsafe_allow_html=True)

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:#9ca3af">
    Built with LangChain · FAISS · Groq (Llama 3.3 70B)<br>
    RAG Chatbot — Portfolio Project
    </div>
    """, unsafe_allow_html=True)


# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📄 PDF Q&A Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a PDF and ask questions — powered by RAG + Llama 3.3 70B</div>', unsafe_allow_html=True)

# Architecture explainer (shown when no PDF is loaded)
if not st.session_state.pdf_processed:
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("1️⃣", "Upload PDF", "PyMuPDF extracts all text from your document"),
        ("2️⃣", "Chunk & Embed", "Text is split into chunks and embedded locally (MiniLM)"),
        ("3️⃣", "Vector Search", "Your question retrieves the most relevant chunks (FAISS)"),
        ("4️⃣", "LLM Answer", "Llama 3.3 70B generates an answer grounded in the context"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(f"""
            <div style="text-align:center;padding:1rem;background:#1e293b;
                        border-radius:10px;border:1px solid #334155;height:140px;color:#e2e8f0;">
                <div style="font-size:2rem">{icon}</div>
                <div style="font-weight:600;margin:0.4rem 0;color:#f1f5f9">{title}</div>
                <div style="font-size:0.8rem;color:#94a3b8">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Enter your API key and upload a PDF to get started!")
    st.stop()

# ── Chat interface ─────────────────────────────────────────────────────────────
st.markdown(f"### 💬 Chat with **{st.session_state.pdf_name}**")
st.markdown("---")

# Render existing chat history
chat_container = st.container()
with chat_container:
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center;padding:2rem;color:#9ca3af">
            <div style="font-size:2rem">💬</div>
            <div>Ask anything about your document!</div>
            <div style="font-size:0.85rem">e.g. "Summarize the main points", "What does section 3 say about X?"</div>
        </div>""", unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""<div class="chat-message-user">
                <strong>You:</strong> {msg["content"]}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="chat-message-bot">
                <strong>🤖 Assistant:</strong> {msg["content"]}
            </div>""", unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("📎 Source excerpts used", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        snippet = src.page_content[:300].strip().replace("\n", " ")
                        st.markdown(f"""<div class="source-box">
                            <strong>Chunk {i}:</strong> …{snippet}…
                        </div>""", unsafe_allow_html=True)

# ── Input ──────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.form("chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "Your question",
            placeholder="Ask a question about the document...",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Send ➤", use_container_width=True, type="primary")

if submitted and user_input.strip():
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        try:
            # Build (user, assistant) history pairs from previous turns only
            history = []
            msgs = st.session_state.chat_history[:-1]  # exclude current question
            i = 0
            while i < len(msgs) - 1:
                if msgs[i]["role"] == "user" and msgs[i+1]["role"] == "assistant":
                    history.append((msgs[i]["content"], msgs[i+1]["content"]))
                    i += 2
                else:
                    i += 1

            result = ask(st.session_state.vector_store, user_input, history, api_key)
            answer = result["answer"]
            sources = result.get("sources", [])
            error = None
        except Exception as e:
            answer = None
            sources = []
            error = str(e)

    if error:
        st.error(f"❌ **Error:** {error}")
        st.session_state.chat_history.pop()  # remove the unanswered user msg
    else:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
        st.rerun()

# Suggested questions
if st.session_state.pdf_processed and not st.session_state.chat_history:
    st.markdown("#### 💡 Try asking:")
    q_cols = st.columns(3)
    sample_qs = [
        "Summarize the key points of this document",
        "What is the main topic discussed?",
        "List the most important facts mentioned",
    ]
    for col, q in zip(q_cols, sample_qs):
        with col:
            if st.button(q, use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("Thinking..."):
                    try:
                        result = ask(st.session_state.vector_store, q, [], api_key)
                        answer = result["answer"]
                        sources = result.get("sources", [])
                    except Exception as e:
                        answer = f"⚠️ Error: {e}"
                        sources = []
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
                st.rerun()
