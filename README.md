# The Archive — Epstein Document Intelligence System

A Retrieval-Augmented Generation (RAG) system for exploring 
publicly released Epstein federal court documents.

## What This Does

- Ingests publicly released PDF court documents
- Cleans, chunks, and embeds them into a vector database
- Accepts natural language questions via a web interface
- Retrieves the most relevant document passages
- Generates factual answers grounded only in the source documents
- Returns source citations for full transparency

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB (persistent) |
| LLM | Llama 3.3 via Groq API (free) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML / CSS / JavaScript |
| OCR | pytesseract + pdf2image |

## Architecture
```
PDF Documents
     ↓
ingest.py → clean_text.py → chunk.py → embed.py
     ↓
ChromaDB (vector store)
     ↓
FastAPI Backend (main.py)
     ↓
Frontend (index.html)
```

## Setup

1. Clone the repository
2. Create virtual environment:
```bash
   cd backend
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
```
3. Add your Groq API key to `backend/.env`:
```
   GROQ_API_KEY=your_key_here
```
4. Run the ingestion pipeline:
```bash
   python ingest.py
   python clean_text.py
   python chunk.py
   python embed.py
```
5. Start the backend:
```bash
   uvicorn main:app --reload
```
6. Open `frontend/index.html` in your browser

## Data Sources

All documents are publicly released federal court records 
from United States v. Maxwell (Case No. 18-CR-217, SDNY)
and related proceedings.

## Disclaimer

This tool is built for research and transparency purposes only.
All source documents are publicly available court records.
```
