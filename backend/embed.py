# embed.py
# Job: Convert text chunks into vectors and store in ChromaDB
# Input:  data/chunked/*.txt
# Output: backend/chroma_data/ (the permanent database)

from sentence_transformers import SentenceTransformer
from pathlib import Path
from database import collection

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
CHUNKED_DIR = BASE_DIR / "data" / "chunked"

# ─────────────────────────────────────────
# LOAD EMBEDDING MODEL
# ─────────────────────────────────────────

# This model converts text into 384 numbers (a vector)
# Similar text = similar numbers = can be searched mathematically
# all-MiniLM-L6-v2 is small, fast, free, and very good
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded.")

# ─────────────────────────────────────────
# EMBEDDING FUNCTION
# ─────────────────────────────────────────

def get_embedding(text):
    """Convert a text string into a list of 384 numbers."""
    return model.encode(text).tolist()

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("\n=== STEP 4: EMBEDDING ===")
    print(f"Reading chunks from: {CHUNKED_DIR}")
    print(f"Saving vectors to:   ChromaDB\n")

    # Check chunk directory exists
    if not CHUNKED_DIR.exists():
        print("❌ Chunk directory not found.")
        print("Please run chunk.py first.")
        return

    # Get all chunk files
    chunk_files = list(CHUNKED_DIR.glob("*.txt"))

    if not chunk_files:
        print("❌ No chunk files found in data/chunked/")
        print("Please run chunk.py first.")
        return

    print(f"Found {len(chunk_files)} chunk files.\n")

    success = 0
    skipped = 0

    for chunk_path in chunk_files:

        # Read the chunk text
        with open(chunk_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Skip empty files
        if not text:
            print(f"⚠️  Skipping empty file: {chunk_path.name}")
            skipped += 1
            continue

        # Check if already in database (avoid duplicates)
        existing = collection.get(ids=[chunk_path.name])
        if existing["ids"]:
            print(f"⏭️  Already embedded: {chunk_path.name}")
            skipped += 1
            continue

        print(f"🔄 Embedding: {chunk_path.name}")

        # Convert text to vector
        embedding = get_embedding(text)

        # Figure out which original PDF this came from
        # filename format: documentname_cleaned_chunk_0.txt
        # we want:         documentname.pdf
        source_name = chunk_path.name.split("_cleaned_chunk_")[0] + ".pdf"

        # Store in ChromaDB with:
        # - documents = the actual text
        # - embeddings = the vector (numbers)
        # - metadatas = which PDF it came from
        # - ids = unique name for this chunk
        collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[{"source": source_name}],
            ids=[chunk_path.name]
        )

        success += 1

    print(f"\n=== EMBEDDING COMPLETE ===")
    print(f"✅ Newly embedded: {success} chunks")
    print(f"⏭️  Skipped:        {skipped} chunks")
    print(f"📊 Total in database: {collection.count()} chunks\n")


if __name__ == "__main__":
    main()