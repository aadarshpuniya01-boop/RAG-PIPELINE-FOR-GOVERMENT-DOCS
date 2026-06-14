import re

from legal_rag.schemas import DocumentType


TYPE_PATTERNS: list[tuple[DocumentType, re.Pattern[str]]] = [
    (DocumentType.OFFICE_MEMORANDUM, re.compile(r"\boffice\s+memorandum\b|\bOM\s+No\.?", re.I)),
    (DocumentType.NOTIFICATION, re.compile(r"\bnotification\b|\bS\.O\.\s*\d+|\bG\.S\.R\.\s*\d+", re.I)),
    (DocumentType.CIRCULAR, re.compile(r"\bcircular\b|\bcircular\s+no\.?", re.I)),
    (DocumentType.GAZETTE, re.compile(r"\bgazette\b|\bextraordinary\b", re.I)),
    (DocumentType.AMENDMENT, re.compile(r"\bamendment\b|\bamended\b|\bpartial\s+modification\b", re.I)),
    (DocumentType.ORDER, re.compile(r"\border\b|\border\s+no\.?", re.I)),
    (DocumentType.GUIDELINE, re.compile(r"\bguidelines?\b|\bstandard\s+operating\s+procedure\b", re.I)),
]


def classify_document(text: str, filename: str = "") -> DocumentType:
    haystack = f"{filename}\n{text[:5000]}"
    scores: dict[DocumentType, int] = {}
    for doc_type, pattern in TYPE_PATTERNS:
        scores[doc_type] = len(pattern.findall(haystack))
    best_type, best_score = max(scores.items(), key=lambda item: item[1])
    return best_type if best_score > 0 else DocumentType.UNKNOWN
