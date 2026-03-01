# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq
from database import collection
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="The Archive API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model ready.")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("✅ Groq client ready.")

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    retrieved_chunks: list[str]

@app.get("/")
def root():
    return {
        "status": "The Archive is operational",
        "documents_indexed": collection.count()
    }

@app.post("/ask")
def ask_question(body: QuestionRequest):

    question = body.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if len(question) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Question too long. Max 1000 characters."
        )

    print(f"\n📥 Question: {question}")

    print("🔢 Generating embedding...")
    query_embedding = embedding_model.encode(question).tolist()

    print("🔍 Searching archive...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas"]
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    sources = []
    for meta in metadatas:
        if meta and "source" in meta:
            if meta["source"] not in sources:
                sources.append(meta["source"])

    if not sources:
        sources = ["Epstein Court Document Archive"]

    print(f"📄 Found {len(chunks)} chunks from: {sources}")

    context = "\n\n---\n\n".join(chunks)

    prompt = f"""You are an investigative research assistant analyzing 
publicly released court documents related to the Epstein case.

Answer using ONLY the document excerpts provided below.
Do not add any information from outside these documents.
If the answer cannot be found in the documents, clearly say so.
Be factual, precise, and professional. No speculation.

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""

    print("🤖 Generating answer with Groq...")
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        answer = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ Groq error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Groq error: {str(e)}"
        )

    print("✅ Answer generated.")

    return AnswerResponse(
        answer=answer,
        sources=sources,
        retrieved_chunks=chunks
    )
