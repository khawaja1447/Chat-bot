# 📄 PDF Q&A RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF and ask natural language questions about its content.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green?logo=langchain)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red?logo=streamlit)
![Gemini](https://img.shields.io/badge/Gemini-1.5_Flash-orange?logo=google)

## 🚀 Live Demo
> _Deploy link here after deploying to Streamlit Community Cloud_

---

## 🏗️ How It Works

```
PDF Upload → Text Extraction (PyMuPDF)
         → Chunking (RecursiveCharacterTextSplitter, 1000 tokens, 200 overlap)
         → Embedding (Gemini embedding-001)
         → FAISS Vector Store
         → Similarity Search (top-4 chunks)
         → Gemini 1.5 Flash generates grounded answer
         → Answer + source excerpts shown in UI
```

## 🧠 Tech Stack

| Component | Technology |
|-----------|------------|
| **Framework** | LangChain 0.3 |
| **LLM** | Google Gemini 1.5 Flash (free tier) |
| **Embeddings** | Gemini `embedding-001` |
| **Vector Store** | FAISS (local, in-memory) |
| **PDF Parsing** | PyMuPDF (fitz) |
| **UI** | Streamlit |
| **Memory** | LangChain `ConversationBufferMemory` |

## ✨ Features

- 📤 **PDF Upload** — any PDF, any topic
- 🧩 **Smart Chunking** — overlapping windows preserve context across chunk boundaries
- 🔍 **Semantic Search** — finds relevant passages even when wording differs
- 💬 **Conversation Memory** — follow-up questions work naturally
- 📎 **Source Citations** — every answer shows which document chunks were used
- 🎨 **Clean UI** — chat-style interface with expandable source excerpts

---

## 🛠️ Local Setup

### 1. Clone & install
```bash
git clone https://github.com/khawaja1447/Chat-bot.git
cd Chat-bot
pip install -r requirements.txt
```

### 2. Get a free Gemini API key
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click **"Create API key"**
3. Copy the key

### 3. Set the key
```bash
# Option A: .env file (recommended for local dev)
cp .env.example .env
# Then edit .env and paste your key

# Option B: enter it directly in the app sidebar
```

### 4. Run
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501)

---

## ☁️ Deploy for Free on Streamlit Community Cloud

1. Push this repo to GitHub (already done ✅)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → select this repo → `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your_key_here"
   ```
5. Click **Deploy** — live in ~2 minutes!

---

## 📁 Project Structure

```
Chat-bot/
├── app.py              # Streamlit UI
├── rag_engine.py       # RAG pipeline (PDF → FAISS → LangChain chain)
├── requirements.txt    # Dependencies
├── .env.example        # API key template
├── .gitignore
└── README.md
```

## 🔑 Key Concepts Demonstrated

| Concept | Where |
|---------|-------|
| **RAG architecture** | `rag_engine.py` — full pipeline |
| **Embeddings** | `build_vector_store()` — Gemini embedding-001 |
| **Vector similarity search** | FAISS `as_retriever(k=4)` |
| **Prompt engineering** | `SYSTEM_PROMPT` in `rag_engine.py` |
| **Conversation memory** | `ConversationBufferMemory` |
| **LLM API integration** | `ChatGoogleGenerativeAI` |
| **Streamlit state management** | `st.session_state` pattern |

---

## 📄 License
MIT
