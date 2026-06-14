from pathlib import Path

import fitz
import pytesseract
from PIL import Image

from legal_rag.config import get_settings


def configure_tesseract() -> None:
    settings = get_settings()
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


def ocr_pdf_pages(path: str | Path, dpi: int = 250) -> list[str]:
    configure_tesseract()
    doc = fitz.open(str(path))
    pages: list[str] = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(image, config="--psm 6")
        pages.append(text)
    return pages
