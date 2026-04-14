from __future__ import annotations

from uuid import uuid4

import requests
import streamlit as st

API_REQUEST_TIMEOUT = 240
INGEST_REQUEST_TIMEOUT = 600
API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(page_title="NUST Bank QA", page_icon="🏦", layout="wide")

st.title("NUST Bank QA")
st.caption("Ask banking questions, review retrieved context, and add new knowledge from the sidebar.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

base_url = API_URL.rsplit("/query", 1)[0] if "/query" in API_URL else API_URL


st.sidebar.markdown("### Add new knowledge")
knowledge_mode = st.sidebar.radio(
    "Choose a method",
    ["Manual entry", "Excel upload"],
    horizontal=True,
    label_visibility="collapsed",
)

if knowledge_mode == "Manual entry":
    st.sidebar.markdown("**Add a single Q&A item**")
    with st.sidebar.form("add_manual_knowledge_form", clear_on_submit=True):
        new_question = st.text_area(
            "Question",
            height=90,
            placeholder="Example: What are the charges for NUST Freelancer Digital account?",
        )
        new_answer = st.text_area(
            "Answer",
            height=140,
            placeholder="Enter the answer exactly as you want it stored in the knowledge base.",
        )
        new_product = st.text_input("Product", value="Manual Entry")
        new_sheet = st.text_input("Source label", value="Manual Input")
        add_submit = st.form_submit_button("Add manual knowledge")

    if add_submit:
        if not new_question.strip() or not new_answer.strip():
            st.sidebar.error("Please provide both a question and an answer.")
        else:
            try:
                ingest_response = requests.post(
                    f"{base_url}/ingest",
                    json={
                        "items": [
                            {
                                "question": new_question,
                                "answer": new_answer,
                                "product": new_product,
                                "sheet": new_sheet,
                            }
                        ]
                    },
                    timeout=INGEST_REQUEST_TIMEOUT,
                )
                ingest_response.raise_for_status()
                ingest_data = ingest_response.json()
            except requests.RequestException as exc:
                st.sidebar.error(f"Ingest failed: {exc}")
            else:
                st.sidebar.success(
                    f"Added {ingest_data.get('added', 0)} item(s). Duplicates skipped: {ingest_data.get('skipped_duplicates', 0)}."
                )
                st.sidebar.info("The API and vector store were refreshed automatically.")
else:
    st.sidebar.markdown("**Upload an Excel workbook**")
    uploaded_excel = st.sidebar.file_uploader(
        "Choose an Excel file",
        type=["xlsx", "xlsm", "xltx", "xltm"],
        help="Workbook should contain FAQ-style content that can be extracted into question and answer pairs.",
    )
    st.sidebar.caption("Supported formats: .xlsx, .xlsm, .xltx, .xltm")

    if st.sidebar.button("Import workbook"):
        if uploaded_excel is None:
            st.sidebar.error("Please choose an Excel workbook first.")
        else:
            try:
                ingest_response = requests.post(
                    f"{base_url}/ingest_excel",
                    files={
                        "file": (
                            uploaded_excel.name,
                            uploaded_excel.getvalue(),
                            uploaded_excel.type or "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    },
                    timeout=INGEST_REQUEST_TIMEOUT,
                )
                ingest_response.raise_for_status()
                ingest_data = ingest_response.json()
            except requests.RequestException as exc:
                st.sidebar.error(f"Excel ingest failed: {exc}")
            else:
                st.sidebar.success(
                    f"Imported {ingest_data.get('filename', uploaded_excel.name)}. "
                    f"Extracted {ingest_data.get('extracted', 0)} pair(s), added {ingest_data.get('added', 0)}."
                )
                st.sidebar.info("The API and vector store were refreshed automatically.")

st.sidebar.markdown("---")

show_context = st.sidebar.checkbox("Show retrieved context", value=True)
use_memory = st.sidebar.checkbox("Use session memory", value=True)
st.sidebar.caption("Session memory is handled automatically in the app. You do not need to manage a session ID manually.")

if st.sidebar.button("Reset memory"):
    try:
        requests.delete(f"{base_url}/memory/{st.session_state.session_id}", timeout=15)
    except requests.RequestException as exc:
        st.sidebar.error(f"Failed to clear memory: {exc}")
    else:
        st.session_state.messages = []
        st.sidebar.success("Memory cleared for this session.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask a banking question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.write("Thinking...")
    with st.spinner("Getting answer..."):
        try:
            response = requests.post(
                API_URL,
                json={
                    "query": question,
                    "k": 3,
                    "session_id": st.session_state.session_id,
                    "use_memory": use_memory,
                },
                timeout=API_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except requests.Timeout:
            answer = (
                "API request timed out. The model may still be loading or generating. "
                "Try a shorter question and confirm the API health endpoint is up."
            )
            placeholder.error(answer)
        except requests.RequestException as exc:
            answer = f"API request failed: {exc}"
            placeholder.error(answer)
        else:
            answer = data.get("answer", "No answer returned.")
            st.session_state.session_id = data.get("session_id", st.session_state.session_id)
            placeholder.write(answer)

            if show_context:
                contexts = data.get("contexts", [])
                if contexts:
                    for i, chunk in enumerate(contexts, start=1):
                        with st.expander(f"Chunk {i}"):
                            st.write(chunk)

    st.session_state.messages.append({"role": "assistant", "content": answer})
