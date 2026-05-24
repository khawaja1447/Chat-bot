"""
RAG Engine — handles PDF ingestion, vector indexing, and retrieval-augmented generation.
"""

import os
import fitz  # PyMuPDF
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

load_dotenv()


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant that answers questions based strictly on
the provided document context. If the answer is not in the context, say
"I couldn't find that in the document." Do not make up information.

Context:
{context}

Question: {question}

Answer (be concise and cite the relevant part of the document when helpful):""",
)


# ── PDF Parsing ────────────────────────────────────────────────────────────────

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract raw text from a Streamlit UploadedFile object."""
    raw_bytes = uploaded_file.read()
    doc = fitz.open(stream=raw_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """Split document text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_text(text)


# ── Vector Store ───────────────────────────────────────────────────────────────

def build_vector_store(chunks: list[str]) -> FAISS:
    """Embed chunks and store in a FAISS index."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store


# ── Conversation Chain ─────────────────────────────────────────────────────────

def build_conversation_chain(vector_store: FAISS) -> ConversationalRetrievalChain:
    """Wrap the retriever in a conversational chain with memory."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
        convert_system_message_to_human=True,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": SYSTEM_PROMPT},
        return_source_documents=True,
        verbose=False,
    )
    return chain


# ── Top-level helper ───────────────────────────────────────────────────────────

def process_pdf(uploaded_file):
    """Full pipeline: PDF → chunks → FAISS → chain. Returns chain."""
    text = extract_text_from_pdf(uploaded_file)
    if not text.strip():
        raise ValueError("No text could be extracted from this PDF.")
    chunks = chunk_text(text)
    vector_store = build_vector_store(chunks)
    chain = build_conversation_chain(vector_store)
    return chain, len(chunks)
