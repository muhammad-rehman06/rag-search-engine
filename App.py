import os
import hashlib
import tempfile

import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚"
)

st.title("📚 PDF RAG Chatbot")
st.write("Upload a PDF and ask questions about it.")


# -----------------------------------
# Helpers
# -----------------------------------

def file_hash(file_bytes: bytes) -> str:
    """Unique id for a file's contents, used as the Chroma collection name.
    This is the fix for the old bug where every upload shared the same
    collection/persist_directory and could leak chunks between documents."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def build_vectorstore(file_bytes: bytes, collection_id: str):
    """Load, chunk and embed a PDF into an isolated, in-memory Chroma
    collection. No persist_directory is used on purpose: each collection
    lives only in this session's memory, so two users (or two uploads in
    the same session) can never see each other's data."""

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        if not docs:
            raise ValueError("No extractable text found in this PDF (it may be a scanned image).")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)

        embedding_model = MistralAIEmbeddings(model="mistral-embed-2312")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name=collection_id,
            # no persist_directory -> ephemeral, in-memory, per-session only
        )

        return vectorstore, len(chunks)

    finally:
        # always clean up the temp file, even if something above failed
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def extract_text(content) -> str:
    """Different LLM providers shape response.content differently.
    Mistral (and most models) return a plain string. Gemini can return a
    list of content blocks like {"type": "text", "text": "...", "extras": {...}}
    instead. This normalizes either shape into plain text so the UI never
    dumps raw Python objects on screen."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()

    return str(content)


PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant answering questions about a specific document.
        Use ONLY the provided context to answer. If the answer is not present in the context,
        say: "I could not find the answer in the document." Do not make up information.

        Recent conversation (for follow-up questions):
        {history}

        Context from the document:
        {context}
    """
    ),
    (
        "human",
        "{question}"
    )
])


# -----------------------------------
# Session state
# -----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ..., "sources": [...] }

if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# -----------------------------------
# Upload
# -----------------------------------

uploaded_file = st.file_uploader("Upload your book/PDF", type=["pdf"])

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    new_id = file_hash(file_bytes)

    # Only reprocess if this is actually a different file than last time.
    # This is what prevents stale-data leakage between different uploads.
    if new_id != st.session_state.current_file_id:
        with st.spinner("Processing PDF..."):
            try:
                vectorstore, n_chunks = build_vectorstore(file_bytes, new_id)
                st.session_state.vectorstore = vectorstore
                st.session_state.current_file_id = new_id
                st.session_state.messages = []  # fresh conversation for a fresh document
                st.success(f"PDF processed successfully ({n_chunks} chunks indexed).")
            except Exception as e:
                st.session_state.vectorstore = None
                st.session_state.current_file_id = None
                st.error(f"Couldn't process this PDF: {e}")

    vectorstore = st.session_state.vectorstore

    if vectorstore is not None:

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
        )

        try:
            llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
        except Exception as e:
            st.error(f"Couldn't initialize the model: {e}")
            st.stop()

        # -----------------------------------
        # Show existing conversation
        # -----------------------------------
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("sources"):
                    with st.expander("View sources"):
                        for i, src in enumerate(msg["sources"], start=1):
                            page = src.metadata.get("page", "unknown")
                            st.markdown(f"**Source {i} (page {page})**")
                            st.text(src.page_content[:400] + ("..." if len(src.page_content) > 400 else ""))

        # -----------------------------------
        # New question
        # -----------------------------------
        query = st.chat_input("Ask something about the PDF...")

        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        docs = retriever.invoke(query)
                        context = "\n\n".join(doc.page_content for doc in docs)

                        # last couple of exchanges, so follow-up questions have context
                        history_pairs = st.session_state.messages[-6:-1]
                        history_text = "\n".join(
                            f"{m['role']}: {m['content']}" for m in history_pairs
                        ) or "None yet."

                        final_prompt = PROMPT.invoke({
                            "context": context,
                            "question": query,
                            "history": history_text
                        })

                        response = llm.invoke(final_prompt)
                        answer = extract_text(response.content)

                        st.write(answer)

                        with st.expander("View sources"):
                            for i, doc in enumerate(docs, start=1):
                                page = doc.metadata.get("page", "unknown")
                                st.markdown(f"**Source {i} (page {page})**")
                                st.text(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": docs
                        })

                    except Exception as e:
                        error_msg = f"Something went wrong while answering: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

else:
    st.session_state.current_file_id = None
    st.session_state.vectorstore = None