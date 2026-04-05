from __future__ import annotations

from uuid import uuid4

import requests
import streamlit as st

API_URL = st.sidebar.text_input("API URL", "http://127.0.0.1:8000/query")

st.title("NUST Bank QA")
st.caption("Chat with your RAG API using session memory.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

show_context = st.sidebar.checkbox("Show retrieved context", value=True)
use_memory = st.sidebar.checkbox("Use session memory", value=True)
request_timeout = st.sidebar.number_input(
    "API timeout (seconds)",
    min_value=30,
    max_value=600,
    value=240,
    step=30,
)
ingest_timeout = st.sidebar.number_input(
    "Ingest timeout (seconds)",
    min_value=30,
    max_value=1800,
    value=600,
    step=30,
)
st.sidebar.text_input("Session ID", value=st.session_state.session_id, disabled=True)

base_url = API_URL.rsplit("/query", 1)[0] if "/query" in API_URL else API_URL

with st.sidebar.expander("Add new knowledge", expanded=False):
    with st.form("add_knowledge_form", clear_on_submit=True):
        uploaded_excel = st.file_uploader(
            "Or upload an Excel workbook",
            type=["xlsx", "xlsm", "xltx", "xltm"],
        )
        new_question = st.text_area("Question", height=100, placeholder="Enter the new question")
        new_answer = st.text_area("Answer", height=160, placeholder="Enter the new answer")
        new_product = st.text_input("Product", value="Manual Entry")
        new_sheet = st.text_input("Sheet", value="Manual Input")
        add_submit = st.form_submit_button("Add to knowledge base")

    if add_submit:
        if uploaded_excel is not None:
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
                    timeout=ingest_timeout,
                )
                ingest_response.raise_for_status()
                ingest_data = ingest_response.json()
            except requests.Timeout:
                st.sidebar.error("Excel ingest timed out. Try a smaller workbook or higher timeout.")
            except requests.RequestException as exc:
                st.sidebar.error(f"Excel ingest failed: {exc}")
            else:
                st.sidebar.success(
                    f"Uploaded {ingest_data.get('filename', uploaded_excel.name)}. "
                    f"Extracted {ingest_data.get('extracted', 0)} pairs, added {ingest_data.get('added', 0)}."
                )
                st.sidebar.info("The API/vectorstore were reloaded automatically.")
        elif not new_question.strip() or not new_answer.strip():
            st.sidebar.error("Either upload an Excel file or provide both question and answer.")
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
                    timeout=ingest_timeout,
                )
                ingest_response.raise_for_status()
                ingest_data = ingest_response.json()
            except requests.Timeout:
                st.sidebar.error("Ingest timed out. Try again with a smaller entry or higher timeout.")
            except requests.RequestException as exc:
                st.sidebar.error(f"Ingest failed: {exc}")
            else:
                st.sidebar.success(
                    f"Added {ingest_data.get('added', 0)} new item(s). "
                    f"Duplicates skipped: {ingest_data.get('skipped_duplicates', 0)}."
                )
                st.sidebar.info("The API/vectorstore were reloaded automatically.")

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
                timeout=request_timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.Timeout:
            answer = (
                "API request timed out. The model may still be loading or generating. "
                "Try a shorter question, increase 'API timeout (seconds)', and confirm /health is up."
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
