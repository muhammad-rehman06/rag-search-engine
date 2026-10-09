"""Streamlit frontend. It contains NO AI logic: it only talks to the FastAPI backend."""

import os
import hashlib

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📚")
st.title("📚 PDF RAG Chatbot")
st.write("Upload a PDF and ask questions about it.")


def error_detail(response: requests.Response) -> str:
    """Pull the friendly message out of an API error response."""
    try:
        return response.json().get("detail", "Something went wrong.")
    except Exception:
        return "Something went wrong."


def show_sources(sources):
    with st.expander("View sources"):
        for i, src in enumerate(sources, start=1):
            st.markdown(f"**Source {i} (page {src.get('page', 'unknown')})**")
            text = src["text"]
            st.text(text + ("..." if len(text) >= 400 else ""))


# -----------------------------------
# Session state
# -----------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_key" not in st.session_state:
    st.session_state.file_key = None
if "document_id" not in st.session_state:
    st.session_state.document_id = None


# -----------------------------------
# Upload
# -----------------------------------
uploaded_file = st.file_uploader("Upload your book/PDF", type=["pdf"])

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_key = hashlib.sha256(file_bytes).hexdigest()

    if file_key != st.session_state.file_key:
        with st.spinner("Processing PDF..."):
            try:
                resp = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded_file.name, file_bytes, "application/pdf")},
                    timeout=300,
                )
                if resp.ok:
                    st.session_state.document_id = resp.json()["document_id"]
                    st.session_state.file_key = file_key
                    st.session_state.messages = []
                    st.success("PDF processed successfully.")
                else:
                    st.session_state.document_id = None
                    st.session_state.file_key = None
                    st.error(f"Couldn't process this PDF: {error_detail(resp)}")
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the API server. Is uvicorn running?")
            except requests.exceptions.Timeout:
                st.error("The server took too long to process this PDF.")

    if st.session_state.document_id:
        # Show existing conversation
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("sources"):
                    show_sources(msg["sources"])

        # New question
        query = st.chat_input("Ask something about the PDF...")

        if query:
            # history = previous messages only (the new question is sent separately)
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-5:]
            ]

            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/ask",
                            json={
                                "document_id": st.session_state.document_id,
                                "question": query,
                                "history": history,
                            },
                            timeout=120,
                        )
                        if resp.ok:
                            data = resp.json()
                            st.write(data["answer"])
                            show_sources(data["sources"])
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": data["answer"],
                                "sources": data["sources"],
                            })
                        elif resp.status_code == 404:
                            # server restarted and forgot the document
                            st.session_state.document_id = None
                            st.session_state.file_key = None
                            st.warning("The server no longer has this document. Please upload it again.")
                        else:
                            st.error(error_detail(resp))
                    except requests.exceptions.ConnectionError:
                        st.error("Can't reach the API server. Is uvicorn running?")
                    except requests.exceptions.Timeout:
                        st.error("The request took too long. Please try again.")
else:
    st.session_state.file_key = None
    st.session_state.document_id = None