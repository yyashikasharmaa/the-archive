# database.py
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

BASE_DIR = Path(__file__).parent
CHROMA_DIR = str(BASE_DIR / "chroma_data")

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

default_ef = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="epstein_docs",
    embedding_function=default_ef,
    metadata={"hnsw:space": "cosine"}
)

print(f"✅ ChromaDB connected. Documents: {collection.count()}")