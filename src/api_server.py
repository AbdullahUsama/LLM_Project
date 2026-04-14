from __future__ import annotations

from collections import defaultdict
from io import BytesIO
from uuid import uuid4
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from src.guardrails import (
    contains_sensitive_data,
    default_guardrails,
    has_sufficient_context,
    is_clearly_out_of_scope,
    looks_like_prompt_injection,
)
from src import llm_llama
from src.data_pipeline import ALL_QA_PATH, ingest_new_qa_pairs, load_json_list, extract_qa_pairs_from_workbook

class QueryRequest(BaseModel):
    query: str
    k: int = 3
    session_id: str | None = None
    use_memory: bool = True


class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]
    session_id: str


class IngestItem(BaseModel):
    question: str
    answer: str
    product: str = "Manual Entry"
    sheet: str = "Manual Input"


class IngestRequest(BaseModel):
    items: List[IngestItem]


class IngestResponse(BaseModel):
    added: int
    skipped_duplicates: int
    total_all_qa: int
    raw_faiss_added_chunks: int
    vectorstore_added_documents: int


class ExcelIngestResponse(BaseModel):
    filename: str
    extracted: int
    added: int
    skipped_duplicates: int
    total_all_qa: int
    raw_faiss_added_chunks: int
    vectorstore_added_documents: int


app = FastAPI(title="NUST Bank QA API", version="1.0.0")

_session_memory: dict[str, list[dict[str, str]]] = defaultdict(list)


def _build_retrieval_query(session_id: str, user_query: str, use_memory: bool) -> str:
    """Build a retrieval query with conservative memory use to avoid topic drift.

    For most turns, retrieval should use only the latest user query. For very short,
    referential follow-up questions (e.g., "what about charges for that?"), append the
    last user turn to recover missing subject context.
    """
    if not use_memory:
        return user_query

    history = _session_memory.get(session_id, [])
    if not history:
        return user_query

    q = user_query.lower().strip()
    tokens = q.split()
    referential_markers = {"it", "that", "this", "those", "these", "they", "them", "its"}
    is_short_follow_up = len(tokens) <= 8 and any(t in referential_markers for t in tokens)

    if not is_short_follow_up:
        return user_query

    last_user = history[-1]["user"]
    return f"Previous question: {last_user}\nCurrent question: {user_query}"


def _remember_turn(session_id: str, user_query: str, answer: str) -> None:
    _session_memory[session_id].append({"user": user_query, "assistant": answer})
    # Limit stored turns per session to keep memory bounded.
    if len(_session_memory[session_id]) > 20:
        _session_memory[session_id] = _session_memory[session_id][-20:]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/memory/{session_id}")
def get_memory(session_id: str) -> dict:
    return {"session_id": session_id, "turns": _session_memory.get(session_id, [])}


@app.delete("/memory/{session_id}")
def clear_memory(session_id: str) -> dict:
    _session_memory.pop(session_id, None)
    return {"session_id": session_id, "cleared": True}


@app.post("/query", response_model=QueryResponse)
def query_api(payload: QueryRequest) -> QueryResponse:
    user_query = payload.query.strip()
    session_id = payload.session_id or str(uuid4())

    if not user_query:
        return QueryResponse(answer="Please provide a question.", contexts=[], session_id=session_id)

    if looks_like_prompt_injection(user_query, default_guardrails):
        return QueryResponse(
            answer=default_guardrails.prompt_injection_reply,
            contexts=[],
            session_id=session_id,
        )

    retrieval_query = _build_retrieval_query(session_id, user_query, payload.use_memory)
    vec_db = llm_llama.get_vectorstore()
    qa_chain = llm_llama.get_qa_chain()
    docs = vec_db.similarity_search(retrieval_query, k=payload.k)
    contexts = [doc.page_content for doc in docs]

    if not has_sufficient_context(contexts, default_guardrails):
        return QueryResponse(
            answer=default_guardrails.insufficient_context_reply,
            contexts=[],
            session_id=session_id,
        )

    if is_clearly_out_of_scope(user_query, contexts, default_guardrails):
        return QueryResponse(
            answer=default_guardrails.out_of_domain_reply,
            contexts=[],
            session_id=session_id,
        )

    result = qa_chain.invoke({"query": retrieval_query})
    answer = result["result"] if isinstance(result, dict) and "result" in result else str(result)

    if contains_sensitive_data(answer, default_guardrails):
        answer = default_guardrails.sensitive_output_reply

    _remember_turn(session_id, user_query, answer)
    return QueryResponse(answer=answer, contexts=contexts, session_id=session_id)


@app.post("/ingest", response_model=IngestResponse)
def ingest_api(payload: IngestRequest) -> IngestResponse:
    new_pairs = [item.model_dump() for item in payload.items if item.question.strip() and item.answer.strip()]
    if not new_pairs:
        return IngestResponse(
            added=0,
            skipped_duplicates=0,
            total_all_qa=len(load_json_list(ALL_QA_PATH)),
            raw_faiss_added_chunks=0,
            vectorstore_added_documents=0,
        )

    summary = ingest_new_qa_pairs(new_pairs)
    llm_llama.refresh_rag_resources()

    return IngestResponse(
        added=summary["training"]["added"],
        skipped_duplicates=summary["training"]["skipped_duplicates"],
        total_all_qa=summary["training"]["total_all_qa"],
        raw_faiss_added_chunks=summary["raw_faiss"]["added_chunks"],
        vectorstore_added_documents=summary["vectorstore"]["added_documents"],
    )


@app.post("/ingest_excel", response_model=ExcelIngestResponse)
async def ingest_excel_api(file: UploadFile = File(...)) -> ExcelIngestResponse:
    if not file.filename.lower().endswith((".xlsx", ".xlsm", ".xltx", ".xltm")):
        raise HTTPException(status_code=400, detail="Please upload an Excel workbook file.")

    workbook_bytes = await file.read()
    extracted_pairs = extract_qa_pairs_from_workbook(workbook_bytes, filename=file.filename)

    if not extracted_pairs:
        return ExcelIngestResponse(
            filename=file.filename,
            extracted=0,
            added=0,
            skipped_duplicates=0,
            total_all_qa=len(load_json_list(ALL_QA_PATH)),
            raw_faiss_added_chunks=0,
            vectorstore_added_documents=0,
        )

    summary = ingest_new_qa_pairs(extracted_pairs)
    llm_llama.refresh_rag_resources()

    return ExcelIngestResponse(
        filename=file.filename,
        extracted=len(extracted_pairs),
        added=summary["training"]["added"],
        skipped_duplicates=summary["training"]["skipped_duplicates"],
        total_all_qa=summary["training"]["total_all_qa"],
        raw_faiss_added_chunks=summary["raw_faiss"]["added_chunks"],
        vectorstore_added_documents=summary["vectorstore"]["added_documents"],
    )
