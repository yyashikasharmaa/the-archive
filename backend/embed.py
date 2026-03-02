# embed.py
# Uses ChromaDB's built-in embedding function
# Much lighter on RAM — works on free hosting

import chromadb
from pathlib import Path

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
CHUNKED_DIR = BASE_DIR / "data" / "chunked"
CHROMA_DIR = Path(__file__).parent / "chroma_data"

print(f"Reading chunks from: {CHUNKED_DIR}")
print(f"Saving to ChromaDB:  {CHROMA_DIR}")

# ─────────────────────────────────────────
# CHROMADB WITH BUILT-IN EMBEDDINGS
# ─────────────────────────────────────────

# ChromaDB has its own built-in embedding function
# Uses a tiny model that only needs ~80MB RAM
# Perfect for free hosting

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# Use ChromaDB's default embedding function
# This is much lighter than sentence-transformers
from chromadb.utils import embedding_functions
default_ef = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="epstein_docs",
    embedding_function=default_ef,
    metadata={"hnsw:space": "cosine"}
)

print(f"✅ Connected. Current count: {collection.count()}")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("\n=== EMBEDDING ===\n")

    if not CHUNKED_DIR.exists():
        print("❌ Chunk directory not found. Run chunk.py first.")
        return

    chunk_files = list(CHUNKED_DIR.glob("*.txt"))

    if not chunk_files:
        print("❌ No chunk files found.")
        return

    print(f"Found {len(chunk_files)} chunks.\n")

    success = 0
    skipped = 0

    for chunk_path in chunk_files:
        with open(chunk_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            skipped += 1
            continue

        existing = collection.get(ids=[chunk_path.name])
        if existing["ids"]:
            skipped += 1
            continue

        print(f"🔄 Embedding: {chunk_path.name}")

        source_name = chunk_path.name.split("_cleaned_chunk_")[0] + ".pdf"

        # ChromaDB handles embedding automatically
        collection.add(
            documents=[text],
            metadatas=[{"source": source_name}],
            ids=[chunk_path.name]
        )

        success += 1

    print(f"\n✅ Done! Embedded: {success} | Skipped: {skipped}")
    print(f"📊 Total in database: {collection.count()}")


if __name__ == "__main__":
    main()