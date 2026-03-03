# LLM Project – Bank Product Knowledge QA

A Retrieval-Augmented Generation (RAG) pipeline that answers banking/product FAQs using a local LLM (Qwen 3.5 via Ollama) grounded in a FAISS vector store built from an Excel knowledge base.

## Pipeline Overview

```
Excel KB → format_for_finetuning.py → all_qa_pairs.json
         → embedder.py  (raw FAISS index)
         → embedder_2.py (LangChain FAISS vectorstore)
         → llm.py (RAG QA chain)
```

## File Descriptions

| File | Purpose |
|------|---------|
| **`embedder.py`** | **Stage 1 embedder.** Loads `all_qa_pairs.json`, cleans & normalises text, chunks each QA pair, embeds them with `all-MiniLM-L6-v2` (SentenceTransformers), and writes a raw FAISS index (`faiss_index.bin`) + chunk metadata JSON. |
| **`embedder_2.py`** | **Stage 2 embedder (LangChain-compatible).** Reads the chunk metadata produced by `embedder.py`, wraps each chunk as a LangChain `Document`, and builds a LangChain-format FAISS vectorstore (`data/vectorstore/`). This is the index that `llm.py` loads. |
| **`llm.py`** | **RAG inference.** Loads the LangChain vectorstore and wires it into a `RetrievalQA` chain with `OllamaLLM` (Qwen 3.5 4B). Retrieves the top-3 relevant chunks and generates an answer. |
| **`search.py`** | Interactive CLI search over the raw FAISS index. Useful for quick similarity lookups without invoking the LLM. |
| **`funds_transer_app_features_faq.json`** | Source FAQ data for funds transfer / app features. |
| **`requirements.txt`** | Python dependencies (LangChain, FAISS, SentenceTransformers, Ollama, etc.). |

### `data/` directory

| File | Purpose |
|------|---------|
| `all_qa_pairs.json` | All extracted QA pairs from the Excel knowledge base. |
| `chunk_metadata.json` | Chunk-level metadata written by `embedder.py`. |
| `format_for_finetuning.py` | Parses the source Excel (`NUST Bank-Product-Knowledge.xlsx`) into JSONL fine-tuning formats and `all_qa_pairs.json`. |
| `inspect_data.py` | Utility to inspect sheets/columns in the source Excel file. |
| `finetuning_data.jsonl` | Instruction-format fine-tuning data. |
| `finetuning_data_chat.jsonl` | OpenAI chat-format fine-tuning data. |
| `vectorstore/` | LangChain FAISS index (`index.faiss` + pickle) used by `llm.py`. |

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build embeddings (run in order)
python embedder.py      # creates raw FAISS index + metadata
python embedder_2.py    # creates LangChain vectorstore

# 3. Start Ollama (separate terminal)
ollama serve

# 4. Run RAG QA
python llm.py
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) with the `qwen3.5:4b` model pulled
