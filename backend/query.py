"""Command-line semantic retrieval test for The Archive.

Uses the same ChromaDB collection and embedding function as the production API
so local retrieval tests reflect application behaviour.
"""

from database import collection


def query_documents(question: str, n_results: int = 3):
    """Return the most relevant document chunks for a natural-language query."""
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return (
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )


if __name__ == "__main__":
    question = input("Query the archive: ").strip()
    if not question:
        raise SystemExit("Question cannot be empty.")

    chunks, metadatas, distances = query_documents(question)

    print("\nTOP RETRIEVED PASSAGES")
    print("=" * 60)
    for index, (chunk, metadata, distance) in enumerate(
        zip(chunks, metadatas, distances), start=1
    ):
        source = (metadata or {}).get("source", "Unknown")
        similarity = 1 - distance
        print(f"\n[{index}] {source} | similarity={similarity:.3f}")
        print(chunk[:500].strip())
