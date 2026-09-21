from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Iterable, List
from pypdf import PdfReader

# Maximum allowed file size in bytes (25 MB)
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024


@dataclass
class DocumentPage:
    text: str
    source: str
    page_number: int


def clean_text(text: str) -> str:
    """Normalize whitespace and remove blank lines."""
    return "\n".join(
        " ".join(line.split())
        for line in text.splitlines()
        if line.strip()
    ).strip()


def validate_file(uploaded_file, max_size_mb: int = 25) -> None:
    """
    Validate file extension and file size.
    Raises ValueError if invalid.
    """
    filename = uploaded_file.name.lower()
    if not filename.endswith(".pdf"):
        raise ValueError(f"Invalid file format: '{uploaded_file.name}'. Only PDF files are supported.")

    uploaded_file.seek(0, 2)
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)

    max_bytes = max_size_mb * 1024 * 1024
    if file_size > max_bytes:
        size_in_mb = file_size / (1024 * 1024)
        raise ValueError(
            f"File size limit exceeded: '{uploaded_file.name}' is {size_in_mb:.2f} MB. "
            f"Maximum allowed file size is {max_size_mb} MB."
        )


def extract_pdf(uploaded_file) -> List[DocumentPage]:
    """
    Extract text from every page of a PDF file using pypdf.
    Preserves document filename and page number metadata.
    Safely skips empty pages.
    """
    validate_file(uploaded_file)

    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    reader = PdfReader(BytesIO(file_bytes))
    pages: List[DocumentPage] = []

    for idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)
        if cleaned:
            pages.append(
                DocumentPage(
                    text=cleaned,
                    source=uploaded_file.name,
                    page_number=idx,
                )
            )

    return pages


def load_documents(uploaded_files: Iterable) -> List[DocumentPage]:
    """Extract pages from all provided PDF files."""
    all_pages: List[DocumentPage] = []

    for uploaded_file in uploaded_files:
        pages = extract_pdf(uploaded_file)
        all_pages.extend(pages)

    if not all_pages:
        raise ValueError(
            "No readable text found in the uploaded documents. "
            "Please ensure the PDFs contain extractable text (scanned PDFs without OCR are not supported)."
        )

    return all_pages
