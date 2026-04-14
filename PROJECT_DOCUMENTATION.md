# LLM Project Documentation

This document describes the current state of the project in detail: architecture, data flow, runtime components, ingestion pipeline, guardrails, anonymization, notebooks, and supporting files.

## Project Summary

This repository is a Retrieval-Augmented Generation (RAG) system for NUST Bank knowledge and FAQ answering. It uses local embeddings, FAISS retrieval, a local Ollama-backed LLM, a FastAPI backend, and a Streamlit frontend. It also supports incremental knowledge ingestion from manual Q&A pairs and Excel workbooks.

The main goal is to answer questions about NUST Bank products, services, app features, policies, limits, and related FAQs using the knowledge stored in the project’s data artifacts.

## Core Capabilities

The project currently supports:

1. Question answering from retrieved knowledge chunks.
2. Session memory for short follow-up conversations.
3. Manual knowledge ingestion from the Streamlit sidebar.
4. Excel workbook ingestion and extraction of Q&A pairs.
5. Deduplication of new knowledge before it is stored.
6. Persistent updates to training files and vector stores.
7. Guardrails for prompt injection, out-of-domain questions, insufficient context, and sensitive-output filtering.
8. Ingestion-time anonymization of sensitive identifiers in new knowledge.

## High-Level Architecture

The system flow is:

1. Source knowledge is entered manually or extracted from Excel.
2. The ingestion pipeline cleans, canonicalizes, anonymizes, and deduplicates the data.
3. The data is stored in JSON/JSONL files and embedded into FAISS-based indexes.
4. The API retrieves relevant chunks for each user query.
5. The LLM generates an answer from the retrieved context.
6. The Streamlit app presents the answer and optionally displays the retrieved chunks.

### Runtime Diagram

```text
Knowledge sources
  -> data_pipeline.py
  -> all_qa_pairs.json / finetuning JSONL / FAISS indexes
  -> llm_llama.py
  -> api_server.py
  -> streamlit_app.py
```

## Key Files

### Root Files

- [README.md](README.md) - Project summary and quick-start information.
- [llm.py](llm.py) - Standalone script that loads the vector store and runs a sample RetrievalQA query.
- [embedder.py](embedder.py) - Rebuilds or appends to the raw FAISS index.
- [embedder_2.py](embedder_2.py) - Rebuilds or appends to the LangChain-compatible FAISS vectorstore.
- [search.py](search.py) - Simple interactive semantic search over the raw FAISS index.
- [run_llm.ipynb](run_llm.ipynb) - Notebook used to install dependencies, start Ollama, run the API, and launch Streamlit in Colab.

### src/

- [src/data_pipeline.py](src/data_pipeline.py) - Canonical ingestion pipeline.
- [src/llm_llama.py](src/llm_llama.py) - Runtime RAG configuration and QA chain.
- [src/api_server.py](src/api_server.py) - FastAPI service exposing query and ingest endpoints.
- [src/streamlit_app.py](src/streamlit_app.py) - User-facing chat UI and ingestion controls.
- [src/guardrails.py](src/guardrails.py) - Query and output safety checks.

### data/

- [data/all_qa_pairs.json](data/all_qa_pairs.json) - Canonical list of Q&A pairs used as the source of truth for indexing and fine-tuning exports.
- [data/chunk_metadata.json](data/chunk_metadata.json) - Serialized chunk metadata aligned to the raw FAISS index.
- [data/faiss_index.bin](data/faiss_index.bin) - Raw FAISS index used for lower-level semantic search and rebuilding.
- [data/vectorstore/](data/vectorstore/) - LangChain FAISS vectorstore loaded by the runtime QA chain.
- [data/finetuning_data.jsonl](data/finetuning_data.jsonl) - Instruction-style fine-tuning file.
- [data/finetuning_data_chat.jsonl](data/finetuning_data_chat.jsonl) - Chat-style fine-tuning file.

### Supporting Data Scripts

- [data/format_for_finetuning.py](data/format_for_finetuning.py) - Converts source knowledge into training-friendly formats.
- [data/inspect_data.py](data/inspect_data.py) - Inspects Excel workbook structure and sheet metadata.

## Data Model

Each knowledge record is treated as a Q&A pair with these fields:

- `question`
- `answer`
- `product`
- `sheet`

The canonical text used for retrieval is generally formatted as:

```text
Question: ...
Answer: ...
```

This same shape is used for:

1. Raw FAISS chunks.
2. LangChain `Document` objects.
3. Fine-tuning records.

## Ingestion Pipeline

The ingestion pipeline is implemented in [src/data_pipeline.py](src/data_pipeline.py).

### Ingestion Sources

The system can ingest knowledge from:

1. Manual Q&A entries submitted from the Streamlit sidebar.
2. Excel workbooks uploaded from the Streamlit sidebar.
3. Any programmatic caller that posts to the FastAPI ingestion endpoints.

### Ingestion Steps

When new knowledge is added, the pipeline performs the following steps:

1. Clean and normalize text.
2. Anonymize sensitive identifiers in the new text.
3. Canonicalize the item into a consistent schema.
4. Deduplicate using a normalized QA signature.
5. Append to the master Q&A JSON file.
6. Append to instruction-style and chat-style JSONL training files.
7. Update the raw FAISS index and chunk metadata.
8. Update the LangChain vectorstore.
9. Refresh the runtime cache so retrieval sees the new knowledge immediately.

### Cleaning

The pipeline performs non-destructive text normalization:

- Trims whitespace.
- Replaces tabs with spaces.
- Collapses multiple spaces.
- Collapses extra blank lines.

### Anonymization

The current ingestion pipeline includes anonymization for new records only.

It masks patterns such as:

- Email addresses.
- Phone numbers.
- CNIC-like numbers.
- Card numbers.
- Account-number-like strings.
- IBAN-like identifiers.

This is controlled by the environment variable `ANONYMIZE_ON_INGEST` and is enabled by default for new ingests.

Important behavior:

1. Existing stored data is not rewritten automatically.
2. Anonymization is applied before deduplication and before storage.
3. The schema is preserved, so downstream code keeps working.

### Deduplication

Deduplication is based on a normalized signature built from:

- question
- answer
- product
- sheet

This prevents duplicate knowledge items from being appended multiple times.

## Vector Storage

The project maintains two retrieval-related artifacts:

### Raw FAISS Index

Stored at [data/faiss_index.bin](data/faiss_index.bin).

This index is built from embedding vectors produced by `SentenceTransformer` and is paired with [data/chunk_metadata.json](data/chunk_metadata.json).

Purpose:

- Low-level semantic search.
- Incremental append/rebuild support.
- Source for inspection/debugging.

### LangChain Vectorstore

Stored under [data/vectorstore/](data/vectorstore/).

This is the runtime vectorstore loaded by [src/llm_llama.py](src/llm_llama.py).

Purpose:

- Retrieval for the QA chain.
- Immediate use by the API and notebook runtime.

## Runtime QA Pipeline

The runtime LLM configuration lives in [src/llm_llama.py](src/llm_llama.py).

### Components

1. `HuggingFaceEmbeddings` with `all-MiniLM-L6-v2`.
2. LangChain FAISS vectorstore loaded from disk.
3. Ollama LLM (`llama3.2:3B` in the runtime module).
4. `RetrievalQA` chain with a prompt template.

### Runtime Behavior

The chain:

1. Retrieves the top-k relevant chunks.
2. Injects those chunks into the prompt as context.
3. Produces an answer constrained by the system prompt.

### Prompt Behavior

The prompt is designed to:

- Stay on NUST Bank topics.
- Prefer evidence from the provided context.
- Avoid speculation.
- Ask for clarification when the question is ambiguous.
- Return a complete answer rather than stopping early.

## API Layer

The FastAPI application is defined in [src/api_server.py](src/api_server.py).

### Endpoints

- `GET /health`
  - Returns a simple health status payload.

- `POST /query`
  - Accepts a query, retrieves context, runs the QA chain, and returns the answer.

- `POST /ingest`
  - Accepts manual Q&A pairs and appends them to the knowledge base.

- `POST /ingest_excel`
  - Accepts an Excel workbook and extracts Q&A pairs for ingestion.

- `GET /memory/{session_id}`
  - Returns in-memory conversation turns for a session.

- `DELETE /memory/{session_id}`
  - Clears stored memory for that session.

### Query Flow

For each question:

1. Strip and validate the query.
2. Check for prompt-injection patterns.
3. Build a conservative retrieval query.
4. Retrieve candidate chunks from FAISS.
5. Check whether the retrieved context is sufficient.
6. Check whether the question is in domain based on query + context.
7. Run the RetrievalQA chain.
8. Filter sensitive output if needed.
9. Store the turn in session memory.

### Session Memory

Session memory is kept in process memory using a `defaultdict(list)`.

It stores:

- user question
- assistant answer

Memory is bounded to avoid unbounded growth.

The current implementation uses a conservative retrieval strategy:

- Most queries are retrieved using the current question only.
- Very short follow-up questions may be augmented with the previous question to recover context.

This reduces topic drift in retrieval.

## Guardrails

The guardrail logic is defined in [src/guardrails.py](src/guardrails.py).

### Existing Guardrails

1. Prompt-injection detection.
2. Context sufficiency checks.
3. Out-of-domain detection using query/context relevance.
4. Sensitive-output filtering.
5. Numeric consistency checks.

### Domain Logic

The current domain check is not purely keyword-blocklist based.

Instead, it uses:

- tokenization
- stopword filtering
- overlap between query and retrieved context
- soft domain hint terms

This reduces false positives compared to static keyword blocking.

### Sensitive Output Guardrail

The system prevents accidental exposure of sensitive data in answers, such as:

- OTP
- PIN
- CVV/CVC
- passwords
- card/account-like number sequences

## Notebooks

### run_llm.ipynb

The notebook is a Colab-oriented launcher and environment bootstrapper.

It performs the following tasks:

1. Installs Python dependencies.
2. Installs and starts Ollama.
3. Pulls the LLM model.
4. Resolves the project root.
5. Loads environment variables from `.env` if present.
6. Starts the FastAPI app.
7. Starts the Streamlit app.
8. Creates an ngrok tunnel for browser access.

It also contains a helper function for guarded query testing:

- `guardrailed_answer(query, k=5)`

This helper mirrors the API logic by:

- checking prompt injection
- retrieving context
- checking context sufficiency
- checking domain relevance
- validating output for unsupported numbers and sensitive data

## Streamlit UI

The frontend in [src/streamlit_app.py](src/streamlit_app.py) provides:

1. A chat interface.
2. A toggle to show retrieved context.
3. A toggle to use or ignore session memory.
4. A session reset button.
5. Manual knowledge ingestion fields.
6. Excel workbook upload for bulk ingestion.

### Manual Knowledge Entry

The sidebar accepts:

- Question
- Answer
- Product
- Sheet

When submitted, it sends a request to `/ingest`.

### Excel Knowledge Entry

The sidebar can upload an Excel workbook and send it to `/ingest_excel`.

The UI reports:

- number of extracted pairs
- number of newly added pairs
- duplicate skips
- reloaded vectorstore notice

## Excel Extraction

Excel ingestion is handled through [src/data_pipeline.py](src/data_pipeline.py).

The extractor supports:

- merged cells
- question/answer header tables
- multi-row answers
- row-wise FAQ layouts
- skipping index/rate-sheet-like tabs

The workbook extraction is designed to work with the format used in the NUST Bank knowledge source.

## Fine-Tuning Artifacts

The project also writes training-friendly data during ingestion.

### Instruction JSONL

Each record contains:

- instruction
- input
- output
- product

### Chat JSONL

Each record contains a `messages` array with:

- system
- user
- assistant

These files are not used directly for runtime retrieval, but they preserve the knowledge in formats suitable for future model tuning.

## Search and Debug Utilities

### llm.py

This is a standalone test harness that:

- loads the embeddings
- loads the FAISS vectorstore
- builds a RetrievalQA chain
- sends a test question

It is useful for quick local validation of the retrieval pipeline.

### search.py

This is an interactive semantic search script over the raw FAISS index.

It lets you:

- enter queries at the terminal
- inspect top retrieved Q/A matches
- verify embedding quality and chunk relevance

### embedder.py

This script appends or rebuilds the raw FAISS index.

### embedder_2.py

This script appends or rebuilds the LangChain vectorstore.

### data/inspect_data.py

This is a workbook inspection tool used to understand the structure of the source Excel file.

### data/format_for_finetuning.py

This script extracts knowledge from the workbook and JSON FAQ sources, then calls the shared ingestion pipeline.

## Configuration and Environment

### Model Settings

Current runtime uses:

- Ollama model: `llama3.2:3B` in [src/llm_llama.py](src/llm_llama.py)
- Embeddings: `all-MiniLM-L6-v2`

### Notebook/Colab Environment

The notebook expects:

- Ollama installed
- required Python packages installed
- a project directory with `src/` available
- optional `.env` file for tokens and settings
- ngrok token if you want public access

### Ingestion Toggle

Anonymization can be controlled with:

- `ANONYMIZE_ON_INGEST=true` (default)
- `ANONYMIZE_ON_INGEST=false`

## What Changes When You Add New Knowledge

When new knowledge is ingested:

1. The master Q&A file is updated.
2. The fine-tuning JSONL files are appended.
3. The raw FAISS index and metadata are updated.
4. The LangChain vectorstore is updated.
5. The API refreshes its cached runtime resources.

This means new knowledge becomes available to retrieval without restarting the whole project manually, though restarting the API/UI is still a safe operational step depending on your environment.

## Current Limits and Considerations

1. Session memory is in-process only; it is not persisted to disk.
2. Existing stored knowledge is not anonymized retroactively.
3. Retrieval quality depends on how well the source knowledge is structured.
4. The system is designed for NUST Bank topics; unrelated questions are intentionally filtered.
5. The runtime depends on local availability of Ollama and the embedding/vectorstore artifacts.

## Practical Workflow

Typical workflow:

1. Prepare or update knowledge.
2. Ingest through the Streamlit sidebar or ingestion scripts.
3. Confirm vectorstore refresh.
4. Ask questions in the chat UI.
5. Inspect retrieved chunks if needed.
6. Use `search.py` or `llm.py` for debugging.

## File-Level Summary

### Root

- `README.md`: concise overview.
- `llm.py`: direct runtime test script.
- `embedder.py`: raw FAISS builder.
- `embedder_2.py`: LangChain vectorstore builder.
- `search.py`: raw search REPL.
- `run_llm.ipynb`: notebook launcher and test environment.

### src/

- `data_pipeline.py`: ingestion, dedupe, anonymization, indexing, and training file generation.
- `llm_llama.py`: runtime RAG setup.
- `api_server.py`: API endpoints and session memory.
- `streamlit_app.py`: UI and ingest controls.
- `guardrails.py`: policy and relevance checks.

### data/

- `format_for_finetuning.py`: source-to-training conversion.
- `inspect_data.py`: workbook inspection utility.
- `all_qa_pairs.json`: master knowledge base.
- `chunk_metadata.json`: raw index metadata.
- `faiss_index.bin`: raw FAISS index.
- `vectorstore/`: runtime LangChain store.
- `finetuning_data.jsonl`: instruction dataset.
- `finetuning_data_chat.jsonl`: chat dataset.

## Summary

This project is a full local RAG pipeline for NUST Bank knowledge. It includes ingestion, deduplication, anonymization, retrieval, guardrails, API access, a web UI, and notebook-based deployment. The pipeline is designed to allow new knowledge to be added incrementally without changing the overall storage format or runtime API shape.
