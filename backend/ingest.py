# ingest.py
# Job: Read PDF files and extract text using OCR
# Input:  data/raw/*.pdf
# Output: data/processed/*.txt

import os
from pathlib import Path

# We use pypdf2 for text-based PDFs first
# If that fails we fall back to OCR (pytesseract)
import pypdf
from pdf2image import convert_from_path
import pytesseract

# ─────────────────────────────────────────
# PATHS — all relative, work on any computer
# ─────────────────────────────────────────

# Find the root of the project (one level up from backend/)
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Create output folder if it doesn't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# POPPLER PATH — needed for pdf2image on Windows
# ─────────────────────────────────────────

# If you have poppler installed, put its bin path here
# Download from: https://github.com/oschwartz10612/poppler-windows/releases
# If you don't have it, we'll use pypdf only (works for text PDFs)
POPPLER_PATH = r"C:\Users\Admin\Downloads\Release-25.12.0-0 (1)\poppler-25.12.0\Library\bin"

# ─────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────

def extract_with_pypdf(pdf_path):
    """
    Try to extract text directly from PDF.
    Works for PDFs with real selectable text.
    Fast and accurate.
    """
    text = ""
    try:
        reader = pypdf.PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n\n--- Page {i+1} ---\n"
                text += page_text
    except Exception as e:
        print(f"  pypdf failed: {e}")
        return ""
    return text.strip()


def extract_with_ocr(pdf_path):
    """
    Convert PDF pages to images then read text using OCR.
    Works for scanned PDFs (images disguised as PDFs).
    Slower but handles scanned documents.
    """
    text = ""
    try:
        pages = convert_from_path(
            pdf_path,
            poppler_path=POPPLER_PATH
        )
        for i, page in enumerate(pages):
            print(f"    OCR processing page {i+1}/{len(pages)}...")
            page_text = pytesseract.image_to_string(page)
            text += f"\n\n--- Page {i+1} ---\n"
            text += page_text
    except Exception as e:
        print(f"  OCR failed: {e}")
        return ""
    return text.strip()


def process_pdf(pdf_path):
    """
    Try pypdf first (fast).
    If it gets less than 100 characters, fall back to OCR (slow but thorough).
    """
    print(f"  Trying direct text extraction...")
    text = extract_with_pypdf(pdf_path)

    if len(text) > 100:
        print(f"  Direct extraction worked. Got {len(text)} characters.")
        return text

    print(f"  Direct extraction got too little text. Trying OCR...")
    text = extract_with_ocr(pdf_path)
    print(f"  OCR complete. Got {len(text)} characters.")
    return text


def main():
    print("\n=== STEP 1: INGESTION ===")
    print(f"Reading PDFs from: {RAW_DIR}")
    print(f"Saving text to:    {PROCESSED_DIR}\n")

    # Get all PDF files
    pdf_files = list(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDF files found in data/raw/")
        print("Please add PDF files and try again.")
        return

    print(f"Found {len(pdf_files)} PDF files.\n")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        # Extract text
        text = process_pdf(pdf_path)

        if not text:
            print(f"  ⚠️  Could not extract text from {pdf_path.name}\n")
            continue

        # Save to processed folder
        output_filename = pdf_path.stem + ".txt"
        output_path = PROCESSED_DIR / output_filename

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"  ✅ Saved to {output_filename}\n")

    print("=== INGESTION COMPLETE ===\n")


if __name__ == "__main__":
    main()