import hashlib
import re
from datetime import date, datetime
from pathlib import Path

from legal_rag.document_classifier import classify_document
from legal_rag.schemas import DocumentType, LegalMetadata


DATE_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%d %B %Y", "%B %d, %Y")


def stable_document_id(source_name: str, text: str) -> str:
    digest = hashlib.sha256(f"{source_name}\n{text[:20000]}".encode("utf-8")).hexdigest()
    return digest[:16]


def _first_match(pattern: str, text: str, flags: int = re.I | re.M) -> str | None:
    match = re.search(pattern, text, flags)
    if not match:
        return None
    return next((group.strip(" :-\n\t") for group in match.groups() if group), match.group(0).strip())


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    cleaned = re.sub(r"\s+", " ", value.strip())
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def _extract_title(text: str, fallback: str) -> str:
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 8]
    for line in lines[:20]:
        if not re.search(r"^(government|ministry|department|no\.|dated|date)\b", line, re.I):
            return line[:220]
    return Path(fallback).stem or "Untitled"


def extract_metadata(text: str, source_name: str, page_count: int, source_url: str | None = None) -> LegalMetadata:
    doc_type = classify_document(text, source_name)
    circular_no = _first_match(r"(?:Circular\s+No\.?|Circular)\s*[:\-]?\s*([A-Z0-9/\-.]+)", text)
    notification_no = _first_match(r"(?:Notification\s+No\.?|Notification)\s*[:\-]?\s*([A-Z0-9/\-.]+)", text)
    issuing_authority = _first_match(r"(?:Issued\s+by|Issuing\s+Authority|Authority)\s*[:\-]\s*(.+)", text)
    department = _first_match(r"(?:Department|Ministry)\s+of\s+([A-Za-z &,]+)", text)
    subject = _first_match(r"(?:Subject|Sub)\s*[:\-]\s*(.+)", text)
    publication = _parse_date(_first_match(r"(?:Publication\s+Date|Published\s+on|Date)\s*[:\-]?\s*([0-9]{1,2}[\/\-.][0-9]{1,2}[\/\-.][0-9]{2,4}|[0-9]{1,2}\s+[A-Za-z]+\s+[0-9]{4}|[A-Za-z]+\s+[0-9]{1,2},\s+[0-9]{4})", text))
    effective = _parse_date(_first_match(r"(?:Effective\s+Date|comes?\s+into\s+force\s+on|with\s+effect\s+from)\s*[:\-]?\s*([0-9]{1,2}[\/\-.][0-9]{1,2}[\/\-.][0-9]{2,4}|[0-9]{1,2}\s+[A-Za-z]+\s+[0-9]{4}|[A-Za-z]+\s+[0-9]{1,2},\s+[0-9]{4})", text))
    version = _first_match(r"(?:Version|Revision)\s*[:\-]\s*([A-Z0-9.\-]+)", text)
    source = source_url if source_url else None

    if doc_type == DocumentType.CIRCULAR and not circular_no:
        circular_no = _first_match(r"\bNo\.\s*([A-Z0-9/\-.]+)", text)
    if doc_type == DocumentType.NOTIFICATION and not notification_no:
        notification_no = _first_match(r"\b(?:S\.O\.|G\.S\.R\.)\s*([0-9A-Z()/\-.]+)", text)

    superseded_refs = re.findall(
        r"(?:supersedes|replaces|in\s+supersession\s+of|partial\s+modification\s+of)\s+([^.\n;]+)",
        text,
        flags=re.I,
    )

    return LegalMetadata(
        document_id=stable_document_id(source_name, text),
        title=_extract_title(text, source_name),
        circular_number=circular_no,
        notification_number=notification_no,
        issuing_authority=issuing_authority,
        publication_date=publication,
        effective_date=effective,
        subject=subject,
        department=department,
        document_type=doc_type,
        superseded_document_references=[ref.strip() for ref in superseded_refs],
        source_url=source,
        version=version,
        page_count=page_count,
    )
