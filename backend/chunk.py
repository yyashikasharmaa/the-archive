# chunk.py
# Job: Cut long cleaned text into small overlapping chunks
# Input:  data/cleaned/*_cleaned.txt
# Output: data/chunked/*_chunk_N.txt

import os
import tiktoken
from pathlib import Path

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
CHUNKED_DIR = BASE_DIR / "data" / "chunked"

CHUNKED_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# CHUNKING SETTINGS
# ─────────────────────────────────────────

# How many tokens per chunk
# 800 tokens ≈ 600 words ≈ about one page of text
# Not too big (AI gets confused) not too small (loses context)
CHUNK_SIZE = 800

# How many tokens to overlap between chunks
# This prevents answers from being split across chunk boundaries
# Example: if a sentence starts at end of chunk 1,
# it also appears at start of chunk 2
CHUNK_OVERLAP = 100

# ─────────────────────────────────────────
# CHUNKING FUNCTION
# ─────────────────────────────────────────

# cl100k_base is the tokenizer used by modern AI models
encoding = tiktoken.get_encoding("cl100k_base")


def chunk_text(text, source_name):
    """
    Split text into overlapping chunks of CHUNK_SIZE tokens.
    Returns list of chunk dictionaries with text and metadata.
    """
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    chunk_number = 0

    while start < len(tokens):
        # Get tokens for this chunk
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]

        # Convert tokens back to text
        chunk_text_str = encoding.decode(chunk_tokens)

        chunks.append({
            "text": chunk_text_str,
            "source": source_name,
            "chunk_number": chunk_number
        })

        chunk_number += 1

        # Move forward by CHUNK_SIZE minus OVERLAP
        # This creates the overlap between consecutive chunks
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def main():
    print("\n=== STEP 3: CHUNKING ===")
    print(f"Reading from: {CLEANED_DIR}")
    print(f"Saving to:    {CHUNKED_DIR}\n")
    print(f"Chunk size:   {CHUNK_SIZE} tokens")
    print(f"Overlap:      {CHUNK_OVERLAP} tokens\n")

    txt_files = list(CLEANED_DIR.glob("*_cleaned.txt"))

    if not txt_files:
        print("❌ No cleaned files found in data/cleaned/")
        print("Please run clean_text.py first.")
        return

    print(f"Found {len(txt_files)} files to chunk.\n")

    total_chunks = 0

    for txt_path in txt_files:
        print(f"Chunking: {txt_path.name}")

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Use original PDF name as source (remove _cleaned suffix)
        source_name = txt_path.stem.replace("_cleaned", "") + ".pdf"

        chunks = chunk_text(text, source_name)

        print(f"  Created {len(chunks)} chunks")

        # Save each chunk as individual file
        for chunk in chunks:
            output_filename = f"{txt_path.stem}_chunk_{chunk['chunk_number']}.txt"
            output_path = CHUNKED_DIR / output_filename

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(chunk["text"])

        total_chunks += len(chunks)
        print(f"  ✅ Saved {len(chunks)} chunk files\n")

    print(f"=== CHUNKING COMPLETE ===")
    print(f"Total chunks created: {total_chunks}\n")


if __name__ == "__main__":
    main()