from legal_rag.document_classifier import classify_document
from legal_rag.schemas import DocumentType


def test_classifies_office_memorandum() -> None:
    assert classify_document("Office Memorandum\nSubject: reimbursement rules") == DocumentType.OFFICE_MEMORANDUM


def test_classifies_notification() -> None:
    assert classify_document("Notification No. G.S.R. 42(E) dated 01/01/2025") == DocumentType.NOTIFICATION
