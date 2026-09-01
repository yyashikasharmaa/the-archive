"""Document ingestion pipeline.

Extracts text from PDFs in data/raw. Text-native PDFs are processed with
pypdf first; scanned PDFs fall back to OCR through pdf2image + Tesseract.
"""

import os
from pathlib import Path

import pypdf
from pdf2image import convert_from_path
import pytesseract

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Optional on Windows. Leave unset when Poppler is available on PATH.
POPPLER_PATH = os.getenv("POPPLER_PATH") or None


def extract_with_pypdf(pdf_path: Path) -> str:
    """Extract selectable text directly from a PDF."""
    text = ""
    try:
        reader = pypdf.PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text += f"\n\n--- Page {page_number} ---\n{page_text}"
    except Exception as exc:
        print(f"Direct extraction failed: {exc}")
        return ""
    return text.strip()


def extract_with_ocr(pdf_path: Path) -> str:
    """OCR scanned/image-only PDF pages."""
    text = ""
    try:
        kwargs = {"poppler_path": POPPLER_PATH} if POPPLER_PATH else {}
        pages = convert_from_path(pdf_path, **kwargs)
        for page_number, page in enumerate(pages, start=1):
            print(f"OCR page {page_number}/{len(pages)}...")
            page_text = pytesseract.image_to_string(page)
            text += f"\n\n--- Page {page_number} ---\n{page_text}"
    except Exception as exc:
        print(f"OCR failed: {exc}")
        return ""
    return text.strip()


def process_pdf(pdf_path: Path) -> str:
    """Prefer direct extraction and fall back to OCR for image-only PDFs."""
    text = extract_with_pypdf(pdf_path)
    if len(text) > 100:
        return text
    return extract_with_ocr(pdf_path)


def main() -> None:
    print("\n=== DOCUMENT INGESTION ===")
    print(f"Input:  {RAW_DIR}")
    print(f"Output: {PROCESSED_DIR}\n")

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in data/raw/.")
        return

    print(f"Found {len(pdf_files)} PDF files.\n")
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        text = process_pdf(pdf_path)
        if not text:
            print(f"Could not extract text from {pdf_path.name}\n")
            continue

        output_path = PROCESSED_DIR / f"{pdf_path.stem}.txt"
        output_path.write_text(text, encoding="utf-8")
        print(f"Saved: {output_path.name}\n")


if __name__ == "__main__":
    main()
