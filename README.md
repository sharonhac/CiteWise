#  CiteWise

> **Senior Israeli Attorney AI** — A modular, high-precision RAG system built for Israeli law firms.

---

## 📁 Directory Structure

```
CITEWISE/
├── .env                    ← Configuration & secrets
├── .cursorrules            ← AI project rules
├── requirements.txt        ← Python dependencies
├── citewise_db.db          ← Auto-generated Milvus Lite DB
├── data/                   ← Drop PDF/Word documents here
├── ingest/
│   ├── __init__.py
│   ├── load_docs.py        ← PDF/Word loader & cleaner
│   ├── chunker.py          ← Legal chunker + definition extractor
│   └── index.py            ← Milvus indexer + sync engine
├── retrieval/
│   ├── __init__.py
│   ├── hybrid_search.py    ← Semantic + BM25 + FlashRank
│   └── retriever.py        ← Context builder & citation formatter
├── generation/
│   ├── __init__.py
│   ├── prompt.py           ← Hebrew legal prompt templates
│   └── llm.py              ← Multi-provider LLM abstraction
├── api/
│   ├── __init__.py
│   └── app.py              ← FastAPI coordinator
└── ui/
    └── streamlit_app.py    ← Hebrew RTL Streamlit UI
```

---

## ⚙️ Setup

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- Ollama models pulled:
  ```bash
  ollama pull llama3
  ollama pull nomic-embed-text
  ```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure `.env`
Edit `.env` to match your environment. Defaults work out of the box for local Ollama.

### 4. Add documents
Copy your PDF or Word files into the `data/` directory.

---

## 🚀 Running CiteWise

**Start the FastAPI backend** (Terminal 1):
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

**Start the Streamlit UI** (Terminal 2):
```bash
streamlit run ui/streamlit_app.py
```

Open your browser at: **http://localhost:8501**

---

## 🔄 Initial Indexing

After starting the API, trigger a manual sync to index your documents:
- **Via UI**: Click the "🔄 סנכרן עכשיו" button in the sidebar.
- **Via CLI**:
  ```bash
  python -m ingest.index
  ```
- **Via API**:
  ```bash
  curl -X POST http://localhost:8000/sync/blocking
  ```

