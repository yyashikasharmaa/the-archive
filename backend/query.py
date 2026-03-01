# query.py
# Job: Search ChromaDB for chunks relevant to a question
# This is used for testing the search directly
# main.py also does this same logic inside the API

from sentence_transformers import SentenceTransformer
from database import collection

# Load same model used during embedding
# MUST be the same model — different models give different vectors
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded.")


def query_documents(question, n_results=3):
    """
    Search the document database for chunks
    most relevant to the question.

    Returns list of text chunks.
    """
    print(f"\n🔍 Searching for: {question}")

    # Convert question to vector
    query_embedding = model.encode(question).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print(f"✅ Found {len(chunks)} relevant chunks.\n")

    return chunks, metadatas, distances


# ─────────────────────────────────────────
# TEST — run this file directly to test search
# python query.py
# ─────────────────────────────────────────

if __name__ == "__main__":
    question = input("Enter your test question: ")

    chunks, metadatas, distances = query_documents(question)

    print("\n" + "="*50)
    print("TOP RELEVANT CHUNKS")
    print("="*50)

    for i, (chunk, meta, dist) in enumerate(zip(chunks, metadatas, distances)):
        print(f"\n--- Result {i+1} ---")
        print(f"Source:     {meta.get('source', 'Unknown')}")
        print(f"Similarity: {round(1 - dist, 3)}")
        print(f"Text preview: {chunk[:300]}...")
        print()