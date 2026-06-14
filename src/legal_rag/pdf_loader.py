import logging
from pathlib import Path

import pdfplumber

from legal_rag.ocr import ocr_pdf_pages

logger = logging.getLogger(__name__)


def extract_pdf_text(path: str | Path, min_chars_per_page: int = 40) -> list[str]:
    pdf_path = Path(path)
    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text(x_tolerance=1.5, y_tolerance=3) or "")

    if pages and sum(len(page.strip()) for page in pages) / max(len(pages), 1) >= min_chars_per_page:
        return pages

    logger.info("PDF appears scanned or low-text; using OCR for %s", pdf_path)
    return ocr_pdf_pages(pdf_path)
