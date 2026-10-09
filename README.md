# PDF RAG Chatbot

A **Retrieval-Augmented Generation (RAG) chatbot** built with **Streamlit, LangChain, ChromaDB, Mistral embeddings, and Google Gemini**.

The application allows users to upload a PDF document and ask questions about its content. The chatbot retrieves relevant sections from the uploaded document and uses an LLM to generate an answer based **only on the retrieved context**.

##  Features

*  Upload PDF documents
*  Extract text from PDFs using `PyPDFLoader`
*  Split documents into smaller chunks
*  Generate embeddings using Mistral AI
*  Store document embeddings in ChromaDB
*  Retrieve relevant chunks using **MMR (Maximal Marginal Relevance)**
*  Generate answers using **Google Gemini**
*  Interactive chat interface using Streamlit
*  Display the source pages used for each answer
*  Support follow-up questions using conversation history
*  Isolate uploaded documents using a unique file hash
*  Automatically remove temporary PDF files after processing

##  How It Works

The application follows this RAG pipeline:

```text
        Upload PDF
             ↓
       Extract Text
             ↓
       Split into Chunks
             ↓
    Generate Embeddings
       (Mistral AI)
             ↓
        ChromaDB
             ↓
      Retrieve Relevant
           Chunks
             ↓
      Build Prompt with
     Context + History
             ↓
       Google Gemini
             ↓
          Answer
             ↓
       Display Sources
```

##  Technologies Used

* **Python**
* **Streamlit** — Web interface
* **LangChain** — RAG application framework
* **Google Gemini** — LLM for generating answers
* **Mistral AI** — Text embeddings
* **ChromaDB** — Vector database
* **PyPDF** — PDF document loading
* **python-dotenv** — Environment variable management

## Project structure

| File | Purpose |
|---|---|
| `main.py` | FastAPI backend: `/upload`, `/ask` and `/health` endpoints |
| `rag.py` | RAG logic: chunking, embeddings, MMR retrieval, Gemini with Groq fallback |
| `app_ui.py` | Streamlit frontend that calls the API |
| `Dockerfile` | Packages the backend so it runs the same everywhere |
| `legacy/App.py` | Version 1: the original single-file Streamlit app |

### How the project evolved

**Version 1 (`legacy/App.py`)** kept the interface and the AI logic in one
Streamlit file. It works well as a demo, but only that one interface can use it.

**Version 2 (current)** separates the AI logic into a FastAPI backend and puts it
in Docker. Any website, bot or app can now call the same backend, and the Streamlit
page is just one possible frontend.

To run the original version: `streamlit run legacy/App.py`
(install both `requirements.txt` and `requirements-ui.txt` first).

## Quick start (Docker)

Create a `.env` file with your own API keys. Write each line with no spaces
around `=` and no quotes around the value:

```
GEMINI_API_KEY=your_key
MISTRAL_API_KEY=your_key
GROQ_API_KEY=your_key
```

Then run:

```
docker run -p 8000:8000 --env-file .env rehman41/pdf-rag-api
```

Open http://localhost:8000/docs to test the API.

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

##  Environment Variables

Create a `.env` file in the project directory:

```env
GOOGLE_API_KEY=your_google_api_key
MISTRAL_API_KEY=your_mistral_api_key
```

**Never upload your `.env` file or API keys to GitHub.**

Make sure `.env` is included in your `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
```

##  Run the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

The application will open in your browser.

Then:

1. Upload a PDF.
2. Wait for the PDF to be processed.
3. Ask a question about the document.
4. The chatbot retrieves relevant content.
5. Gemini generates an answer using the retrieved context.
6. Expand **View sources** to see the source pages.

##  RAG Implementation

The application divides the PDF into chunks using:

```python
RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
```

Mistral embeddings are then generated for these chunks:

```python
MistralAIEmbeddings(
    model="mistral-embed-2312"
)
```

The embeddings are stored in ChromaDB and retrieved using MMR:

```python
vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)
```

## 🤖 Question Answering

The chatbot is instructed to use only the retrieved document context.

If the answer cannot be found in the document, the model is instructed to respond:

```text
I could not find the answer in the document.
```

This helps reduce unsupported or hallucinated answers.

##  Document Isolation

Each uploaded PDF is assigned a unique identifier using a SHA-256 hash of its contents:

```python
hashlib.sha256(file_bytes).hexdigest()[:16]
```

The application uses this identifier as the Chroma collection name. Each collection is kept in memory for the current session rather than using a persistent directory.

This prevents chunks from different uploaded documents from being mixed together.

##  Conversation History

The application keeps recent conversation messages and passes the last few exchanges to the prompt. This allows the chatbot to understand follow-up questions related to the current document.

##  Source References

For every generated answer, the application displays the retrieved source chunks along with their PDF page numbers.

This makes it easier to verify where the answer came from.

##  Limitations

* The chatbot can only answer questions based on text that can be extracted from the PDF.
* Scanned/image-only PDFs may not work without OCR.
* API keys are required for Gemini and Mistral services.
* Vector data is stored only for the current session.
* Very large PDFs may take longer to process.

##  Future Improvements

Possible improvements include:

*  Support multiple PDFs
*  Add document re-indexing
*  Add persistent vector storage
*  Create a multi-document knowledge base
*  Add OCR support for scanned PDFs
*  Add user authentication
*  Add retrieval and answer evaluation
*  Deploy the application online

##  Author

**Your Name**

Built as a Generative AI / RAG project using Python and LangChain.

##  If You Find This Project Useful

Feel free to fork the repository, experiment with the code, and improve the project.

