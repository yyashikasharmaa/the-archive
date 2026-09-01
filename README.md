# THE ARCHIVE — Grounded Court-Document Intelligence

> A full-stack Retrieval-Augmented Generation (RAG) system that transforms public federal court records into a searchable semantic knowledge base and answers natural-language questions using retrieved evidence rather than unconstrained model memory.

**Live application:** https://the-archive-seven.vercel.app

## Why This Project Exists

Large legal-document collections are difficult to investigate manually: records span many pages, terminology varies, scanned PDFs require OCR, and ordinary keyword search misses semantically related passages.

**The Archive** builds an end-to-end document-intelligence pipeline around that problem. It ingests court PDFs, extracts and cleans their text, divides the corpus into retrieval-sized passages, embeds those passages into a persistent vector database, retrieves evidence semantically, and supplies only the retrieved context to an LLM for a grounded answer.

The result is closer to an **evidence-backed research interface** than a generic chatbot: retrieval happens before generation, answers are constrained to the indexed corpus, and source documents are returned with each response.

## Product Interface

The deployed frontend uses a custom investigative-record aesthetic rather than a stock chat UI. It provides a single-purpose query workflow designed around document interrogation and source-grounded retrieval.

> Repository screenshot will be added under `docs/screenshots/archive-home.png` from the deployed application.

## System Architecture

```mermaid
flowchart TD
    A[Public Court PDFs] --> B{Text available?}
    B -->|Yes| C[pypdf extraction]
    B -->|Scanned| D[PDF rasterization + Tesseract OCR]
    C --> E[Text cleaning]
    D --> E
    E --> F[Document chunking]
    F --> G[Semantic embeddings]
    G --> H[(Persistent ChromaDB Vector Store)]

    I[Natural-language query] --> J[FastAPI /ask endpoint]
    J --> K[Semantic vector retrieval]
    H --> K
    K --> L[Top relevant document passages]
    L --> M[Grounded prompt construction]
    M --> N[Llama 3.3 70B via Groq]
    N --> O[Answer + source documents + retrieved evidence]
    O --> P[Custom web interface]
```

## Intelligence Pipeline

### 1. Hybrid PDF Ingestion

`backend/ingest.py`

The ingestion layer supports both text-native and scanned records:

- attempts direct extraction with `pypdf` first;
- detects insufficient extracted text;
- falls back to page rasterization with `pdf2image`;
- performs OCR with Tesseract;
- preserves page boundaries in the extracted representation;
- uses portable project-relative paths and an optional `POPPLER_PATH` environment variable.

### 2. Corpus Cleaning & Chunking

`backend/clean_text.py` → `backend/chunk.py`

Raw extraction output is normalized before being divided into smaller passages suitable for semantic retrieval. Chunk-level files provide a transparent intermediate representation between source documents and the vector index.

### 3. Persistent Semantic Index

`backend/embed.py`

Processed chunks are embedded into a persistent **ChromaDB** collection configured for cosine-space retrieval. Each vector entry retains source-document metadata, allowing retrieval results to remain traceable to the underlying record.

The embedding stage is incremental: existing chunk IDs are detected and skipped instead of being blindly re-indexed.

### 4. Evidence Retrieval

`backend/query.py`

A query is embedded through the same ChromaDB collection used by the application. The system retrieves the highest-ranking passages together with source metadata and vector distance information.

This separates **retrieval** from **generation**: the language model does not decide which documents are relevant by itself.

### 5. Grounded Generation API

`backend/main.py`

The FastAPI service exposes a `/ask` endpoint that:

1. validates the incoming question;
2. retrieves the top matching passages from ChromaDB;
3. constructs a context window from those passages;
4. instructs the model to use only that evidence and avoid speculation;
5. generates the response using **Llama 3.3 70B through Groq**;
6. returns the answer, unique source-document names, and the retrieved passages.

Generation uses a low temperature (`0.1`) to favour consistent, evidence-oriented responses.

## What Makes It More Than a Chatbot

| Capability | Implementation |
|---|---|
| Semantic search | Vector retrieval instead of literal keyword matching |
| Scanned-document support | OCR fallback for image-only PDFs |
| Persistent knowledge base | ChromaDB vector storage |
| Grounded generation | LLM receives retrieved court excerpts as context |
| Source traceability | Source metadata returned with answers |
| Retrieval transparency | API also returns the passages used for generation |
| Corpus isolation | Prompt explicitly prohibits outside information |
| Full-stack delivery | FastAPI backend + deployed custom web interface |

## Technology Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Pydantic, Uvicorn |
| Vector database | ChromaDB |
| Retrieval | Semantic vector similarity / cosine space |
| Generation | Llama 3.3 70B via Groq API |
| PDF extraction | pypdf |
| OCR fallback | Tesseract OCR + pdf2image |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Vercel frontend + Render-compatible backend configuration |
| Configuration | python-dotenv / environment variables |

## Repository Structure

```text
the-archive/
├── backend/
│   ├── main.py             # FastAPI RAG API
│   ├── ingest.py           # PDF extraction + OCR fallback
│   ├── clean_text.py       # corpus normalization
│   ├── chunk.py            # retrieval-sized passage generation
│   ├── embed.py            # persistent ChromaDB indexing
│   ├── query.py            # direct semantic retrieval test
│   ├── database.py         # vector-store connection
│   ├── requirements.txt
│   └── render.yaml
├── data/
│   ├── raw/                # source records
│   ├── processed/          # extracted/cleaned text
│   └── chunked/            # retrieval passages
├── frontend/
│   ├── index.html          # deployed research interface
│   └── bg.mp4              # interface visual asset
└── README.md
```

## API Contract

### `GET /`

Returns service health information and the number of indexed records in the active ChromaDB collection.

### `POST /ask`

Request:

```json
{
  "question": "What does the record state about ...?"
}
```

Response shape:

```json
{
  "answer": "Grounded answer generated from retrieved passages",
  "sources": ["source-document.pdf"],
  "retrieved_chunks": ["supporting passage ..."]
}
```

## Local Setup

### Backend

```bash
git clone https://github.com/yyashikasharmaa/the-archive.git
cd the-archive/backend

python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

Create `backend/.env`:

```env
GROQ_API_KEY=your_key_here
# Optional on Windows when Poppler is not on PATH:
# POPPLER_PATH=C:\path\to\poppler\Library\bin
```

For rebuilding the source corpus, install the ingestion dependencies (`pypdf`, `pdf2image`, `pytesseract`) and ensure Tesseract/Poppler are available on the host.

Run the preprocessing/indexing pipeline from `backend/`:

```bash
python ingest.py
python clean_text.py
python chunk.py
python embed.py
```

Start the API:

```bash
uvicorn main:app --reload
```

The frontend can then be served from `frontend/` and pointed at the API deployment.

## Engineering Decisions

**Retrieval before generation.** The system searches the corpus first and gives the model only a small set of relevant passages. This reduces dependence on model memory and makes the answer auditable against retrieved evidence.

**Direct extraction before OCR.** OCR is computationally expensive and can introduce recognition errors. Text-native PDFs therefore use direct parsing first, while OCR is reserved for documents that require it.

**Persistent vector storage.** Embeddings are generated during corpus preparation rather than recomputed for every user query, keeping online retrieval lightweight.

**Low-temperature generation.** The application is intended for document research, not creative writing. A low generation temperature is used to prioritize precision and consistency.

**Source metadata at chunk level.** Every indexed passage carries its originating filename so the API can reconstruct source attribution after semantic retrieval.

## Limitations & Next Steps

The current implementation is a focused portfolio-scale RAG system rather than a production legal-research platform. Natural extensions include reranking, page-level citations, hybrid BM25 + vector retrieval, automated RAG evaluation, streaming responses, authentication/rate limiting, and richer corpus metadata.

## Data & Responsible Use

The project is designed around publicly released federal court records and is intended as a document-retrieval/research demonstration. Generated responses should be treated as navigation aids to the underlying records, not as independent factual or legal conclusions. Users should verify important claims against the cited source material.

## Author

**Yashika Sharma**  
Computer Science — Data Science  
AI systems · RAG · robotics · product-oriented engineering
