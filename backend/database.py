# database.py
# Creates a shared ChromaDB connection that saves to disk
# Other files import 'collection' from here

import chromadb
from pathlib import Path

# Path where ChromaDB saves its files
BASE_DIR = Path(__file__).parent
CHROMA_DIR = str(BASE_DIR / "chroma_data")

# Create persistent client (saves to disk)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# Create or connect to collection
collection = chroma_client.get_or_create_collection(
    name="epstein_docs",
    metadata={"hnsw:space": "cosine"}
)

print(f"📂 ChromaDB location: {CHROMA_DIR}")
print(f"✅ Connected to collection: epstein_docs")
print(f"📊 Documents in database: {collection.count()}")