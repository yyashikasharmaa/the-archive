# clean_text.py
# Job: Clean up the messy text extracted from PDFs
# Input:  data/processed/*.txt
# Output: data/cleaned/*_cleaned.txt

import os
import re
from pathlib import Path

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CLEANED_DIR = BASE_DIR / "data" / "cleaned"

CLEANED_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# CLEANING FUNCTION
# ─────────────────────────────────────────

def clean_text(text):
    """
    Clean raw PDF text by removing common issues.

    What each step does:
    1. Remove null bytes (invisible garbage characters)
    2. Fix weird Unicode characters
    3. Remove excessive whitespace
    4. Remove lines that are just page numbers
    5. Remove lines that are just dashes or underscores
    6. Collapse multiple blank lines into one
    """

    # Step 1 — Remove null bytes and control characters
    text = text.replace('\x00', '')
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Step 2 — Normalize unicode quotes and dashes
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')

    # Step 3 — Remove hyphenation at end of lines
    # (PDFs often split words across lines like "testi-\nmony")
    text = re.sub(r'-\n(\w)', r'\1', text)

    # Step 4 — Fix spacing issues
    text = re.sub(r' +', ' ', text)        # Multiple spaces → one space
    text = re.sub(r'\t', ' ', text)         # Tabs → space

    # Step 5 — Process line by line
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Skip empty lines (we'll add them back controlled)
        if not line:
            cleaned_lines.append('')
            continue

        # Skip lines that are just page numbers (1, 2, 23 etc alone on line)
        if re.match(r'^\d{1,3}$', line):
            continue

        # Skip lines that are just dashes, underscores, or dots
        if re.match(r'^[-_=.]{3,}$', line):
            continue

        # Skip very short garbage lines (less than 3 characters)
        if len(line) < 3:
            continue

        cleaned_lines.append(line)

    # Step 6 — Collapse multiple blank lines into maximum 2
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def main():
    print("\n=== STEP 2: CLEANING ===")
    print(f"Reading from: {PROCESSED_DIR}")
    print(f"Saving to:    {CLEANED_DIR}\n")

    txt_files = list(PROCESSED_DIR.glob("*.txt"))

    if not txt_files:
        print("❌ No .txt files found in data/processed/")
        print("Please run ingest.py first.")
        return

    print(f"Found {len(txt_files)} files to clean.\n")

    for txt_path in txt_files:
        print(f"Cleaning: {txt_path.name}")

        with open(txt_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        print(f"  Before: {len(raw_text)} characters")

        cleaned = clean_text(raw_text)

        print(f"  After:  {len(cleaned)} characters")

        # Save with _cleaned suffix
        output_filename = txt_path.stem + "_cleaned.txt"
        output_path = CLEANED_DIR / output_filename

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f"  ✅ Saved {output_filename}\n")

    print("=== CLEANING COMPLETE ===\n")


if __name__ == "__main__":
    main()