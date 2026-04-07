# LLM Project - Bank Product Knowledge QA

A Retrieval-Augmented Generation (RAG) project for NUST Bank FAQs. It uses a local Ollama model, FAISS retrieval, guardrails, a FastAPI backend, and a Streamlit chat app.

## Group Members

- Hissan Umar - 411644
- Abdullah Usama - 417872

## Architecture

```
Knowledge data (JSON/Excel)
    -> data_pipeline.py (clean, dedupe, append)
    -> raw FAISS index + chunk metadata
    -> LangChain FAISS vectorstore (data/vectorstore)
    -> llm_llama.py (retriever + QA chain)
    -> api_server.py (/query, /ingest, /ingest_excel, memory endpoints)
    -> streamlit_app.py (chat + add knowledge UI)
```

## Current Core Files

| File | Purpose |
|------|---------|
| `api_server.py` | FastAPI app that serves `/query`, ingestion endpoints, health checks, and session-memory routes. |
| `streamlit_app.py` | Chat UI that calls the API, shows retrieved chunks, and supports adding new knowledge from text or Excel. |
| `llm_llama.py` | RAG runtime: loads LangChain FAISS, builds RetrievalQA chain, and provides cache refresh after ingestion. |
| `guardrails.py` | Domain and context checks used before answer generation (out-of-domain and insufficient-context handling). |
| `data_pipeline.py` | Incremental ingestion pipeline: canonicalize, dedupe, append training files, update FAISS indexes/vectorstore. |
| `embedder.py` | Entry point to append/rebuild the raw FAISS index from current QA data. |
| `embedder_2.py` | Entry point to append/rebuild LangChain-compatible FAISS vectorstore. |
| `search.py` | CLI semantic search utility over the raw FAISS index. |

## Data Files

| Path | Purpose |
|------|---------|
| `data/all_qa_pairs.json` | Master QA pairs used as the source for indexing. |
| `data/chunk_metadata.json` | Chunk text + metadata aligned with raw FAISS vectors. |
| `data/faiss_index.bin` | Raw FAISS index for low-level similarity search. |
| `data/vectorstore/index.faiss` + `data/vectorstore/index.pkl` | LangChain FAISS vectorstore loaded by runtime RAG code. |
| `data/finetuning_data.jsonl` | Instruction-style fine-tuning records appended during ingestion. |
| `data/finetuning_data_chat.jsonl` | Chat-format fine-tuning records appended during ingestion. |

## Model and Embeddings

- Embeddings: `all-MiniLM-L6-v2`
- Ollama LLM used by runtime: `llama3.2:3B`

Before running, make sure Ollama is installed and the model is pulled.

```bash
ollama pull llama3.2:3B
```

## Local Run (Root Layout)

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Build or refresh retrieval indexes
python embedder.py
python embedder_2.py

# 3) Start Ollama (separate terminal)
ollama serve

# 4) Start API
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

# 5) Start Streamlit (new terminal)
streamlit run streamlit_app.py
```

## Notebook / Colab Run (src Layout)

The notebook `run_llm.ipynb` is now configured to detect and run files from a `src` folder (for example `/content/LLM_Project/src`).

In that setup, keep runtime modules in `src`:

- `src/api_server.py`
- `src/streamlit_app.py`
- `src/llm_llama.py`
- `src/guardrails.py`
- `src/data_pipeline.py`

and ensure required data artifacts are available under `src/data`.

## API Endpoints

- `GET /health` - health check
- `POST /query` - retrieve + generate answer
- `POST /ingest` - add manual QA items
- `POST /ingest_excel` - upload workbook and extract QA pairs
- `GET /memory/{session_id}` - view in-memory turns
- `DELETE /memory/{session_id}` - clear memory for a session

## Streamlit Features

- Chat interface connected to FastAPI `/query`
- Optional session memory toggle
- Optional retrieved-context display
- Sidebar ingestion form:
    - manual question/answer/product/sheet input
    - Excel workbook upload (`.xlsx/.xlsm/.xltx/.xltm`)
- Ingestion status and duplicate-skip reporting

## What Happens During Ingestion

1. New items are cleaned and canonicalized.
2. Duplicates are skipped using a normalized QA signature.
3. `all_qa_pairs.json` is updated.
4. Fine-tuning JSONL files are appended.
5. Raw FAISS and chunk metadata are appended/recovered.
6. LangChain vectorstore is appended/rebuilt.
7. API refreshes cached vectorstore/QA-chain resources so new knowledge is available immediately.

## Requirements

- Python 3.10+
- Ollama installed and running
- Dependencies from `requirements.txt`
