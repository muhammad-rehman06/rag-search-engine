"""RAG logic for the PDF chatbot.
No Streamlit code in this file: it only knows about PDFs, vectors and LLMs,
so it can be used by FastAPI, a script, or anything else.
"""

import os
import hashlib
import tempfile

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

GEMINI_MODEL = "gemini-3.6-flash"   # change here if the model gets renamed or retired
GROQ_MODEL = "openai/gpt-oss-120b"  # backup model

# document_id -> Chroma vectorstore. Lives in server memory, so it is
# cleared when the server restarts (fine for now; persistence comes later).
_VECTORSTORES = {}

_LLM = None  # built once on first use, then reused


class DocumentNotFound(Exception):
    """Raised when a question refers to a document_id the server doesn't have."""


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
# Helpers
# -----------------------------------

def file_hash(file_bytes: bytes) -> str:
    """Unique id for a file's contents, used as document_id and collection name."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def extract_text(content) -> str:
    """Normalize response.content (string or list of blocks) into plain text."""
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


def get_llm():
    """Gemini with retries, falling back to Groq if Gemini keeps failing."""
    global _LLM
    if _LLM is None:
        primary = ChatGoogleGenerativeAI(model=GEMINI_MODEL, max_retries=5)
        backup = ChatGroq(model=GROQ_MODEL, max_retries=3)
        _LLM = primary.with_fallbacks([backup])
    return _LLM


def build_vectorstore(file_bytes: bytes, collection_id: str):
    """Load, chunk and embed a PDF into an isolated in-memory Chroma collection."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        if not docs:
            raise ValueError("No extractable text found in this PDF (it may be a scanned image).")

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        embedding_model = MistralAIEmbeddings(model="mistral-embed-2312")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name=collection_id,
        )
        return vectorstore, len(chunks)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# -----------------------------------
# Public functions (used by main.py)
# -----------------------------------

def process_pdf(file_bytes: bytes) -> dict:
    """Index a PDF and return its document_id.
    If the same file was already processed, reuse it instead of re-embedding."""
    doc_id = file_hash(file_bytes)

    if doc_id in _VECTORSTORES:
        return {"document_id": doc_id, "chunks": None, "already_processed": True}

    vectorstore, n_chunks = build_vectorstore(file_bytes, doc_id)
    _VECTORSTORES[doc_id] = vectorstore
    return {"document_id": doc_id, "chunks": n_chunks, "already_processed": False}


def ask_question(document_id: str, question: str, history: list | None = None) -> dict:
    """Answer a question using only the given document.

    history: list of {"role": "user" | "assistant", "content": "..."}
    Returns: {"answer": str, "pages": [int], "sources": [{"page": int, "text": str}]}
    """
    vectorstore = _VECTORSTORES.get(document_id)
    if vectorstore is None:
        raise DocumentNotFound(f"No document with id '{document_id}'. Upload the PDF first.")

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    history = history or []
    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in history[-5:]
    ) or "None yet."

    final_prompt = PROMPT.invoke({
        "context": context,
        "question": question,
        "history": history_text,
    })

    response = get_llm().invoke(final_prompt)
    answer = extract_text(response.content)

    sources = []
    for doc in docs:
        page = doc.metadata.get("page")
        page = page + 1 if isinstance(page, int) else None  # PyPDF pages start at 0
        sources.append({"page": page, "text": doc.page_content[:400]})

    pages = sorted({s["page"] for s in sources if s["page"] is not None})
    return {"answer": answer, "pages": pages, "sources": sources}