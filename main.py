from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from rag import process_pdf, ask_question, DocumentNotFound

app = FastAPI(title="PDF RAG API")


class ChatMessage(BaseModel):
    role: str      # "user" or "assistant"
    content: str


class Question(BaseModel):
    document_id: str
    question: str
    history: list[ChatMessage] = []


@app.get("/health")
def health():
    return {"status": "ok"}


# These are normal `def` (not `async def`) on purpose: the work is blocking
# (embeddings, LLM calls), so FastAPI runs them in a worker thread instead of
# freezing the whole server while one request is busy.

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        return process_pdf(data)
    except ValueError as e:
        # e.g. scanned PDF with no extractable text
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        print(f"[upload error] {e}")  # full error stays in the server terminal
        raise HTTPException(status_code=502, detail="Could not process this PDF. Please try again.")


@app.post("/ask")
def ask(body: Question):
    try:
        history = [m.model_dump() for m in body.history]
        return ask_question(body.document_id, body.question, history)
    except DocumentNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"[ask error] {e}")  # full error stays in the server terminal
        raise HTTPException(
            status_code=503,
            detail="The AI service is busy right now. Please try again in a moment.",
        )