"""
RAG Engine — PDF ingestion, FAISS vector store, and retrieval-augmented generation.
LLM: Groq (llama-3.3-70b-versatile) | Embeddings: HuggingFace all-MiniLM-L6-v2 (local)
"""

import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from groq import Groq

import fitz  # PyMuPDF
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


# ── PDF Parsing ────────────────────────────────────────────────────────────────

def extract_text_from_pdf(uploaded_file) -> str:
    raw_bytes = uploaded_file.read()
    doc = fitz.open(stream=raw_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_text(text)


# ── Vector Store ───────────────────────────────────────────────────────────────

def build_vector_store(chunks: list) -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_texts(chunks, embedding=embeddings)


# ── Direct Gemini API call ─────────────────────────────────────────────────────

def _call_llm(prompt: str, api_key: str) -> str:
    """Call Groq (llama-3.3-70b) — free tier: 14,400 req/day, 30 req/min."""
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ── ask() — retrieval + generation ────────────────────────────────────────────

def ask(vector_store: FAISS, question: str, history: list, api_key: str) -> dict:
    """
    Retrieve top-4 chunks, build a grounded prompt, call Groq LLM, return answer + sources.
    history: list of (user_str, ai_str) tuples for conversation context.
    """
    docs = vector_store.similarity_search(question, k=4)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    # Build conversation history string
    history_text = ""
    for user_msg, ai_msg in history[-3:]:   # last 3 turns max
        history_text += f"User: {user_msg}\nAssistant: {ai_msg}\n\n"

    prompt = f"""You are a helpful assistant that answers questions based strictly on the document context below.
If the answer is not in the context, say "I couldn't find that in the document." Do not make up information.

DOCUMENT CONTEXT:
{context}

CONVERSATION HISTORY:
{history_text}
User: {question}
Assistant:"""

    answer = _call_llm(prompt, api_key)
    return {"answer": answer, "sources": docs}


# ── Top-level pipeline ─────────────────────────────────────────────────────────

def process_pdf(uploaded_file):
    """PDF → chunks → FAISS. Returns (vector_store, chunk_count)."""
    text = extract_text_from_pdf(uploaded_file)
    if not text.strip():
        raise ValueError("No text could be extracted from this PDF.")
    chunks = chunk_text(text)
    vector_store = build_vector_store(chunks)
    return vector_store, len(chunks)
