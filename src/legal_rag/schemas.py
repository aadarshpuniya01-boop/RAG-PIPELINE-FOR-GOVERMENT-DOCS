from datetime import date
from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel, Field, HttpUrl


class DocumentType(StrEnum):
    CIRCULAR = "Circular"
    NOTIFICATION = "Notification"
    GAZETTE = "Gazette"
    OFFICE_MEMORANDUM = "Office Memorandum"
    AMENDMENT = "Amendment"
    ORDER = "Order"
    GUIDELINE = "Guideline"
    UNKNOWN = "Unknown"


class SupersessionType(StrEnum):
    SUPERSEDES = "supersedes"
    PARTIAL_MODIFICATION = "partial_modification"
    AMENDS = "amends"
    REPLACES = "replaces"


class SupersessionRelation(BaseModel):
    relation_type: SupersessionType
    target_reference: str
    evidence_text: str
    confidence: float = Field(ge=0.0, le=1.0)


class LegalMetadata(BaseModel):
    document_id: str
    title: str = "Untitled"
    circular_number: Optional[str] = None
    notification_number: Optional[str] = None
    issuing_authority: Optional[str] = None
    publication_date: Optional[date] = None
    effective_date: Optional[date] = None
    subject: Optional[str] = None
    department: Optional[str] = None
    document_type: DocumentType = DocumentType.UNKNOWN
    superseded_document_references: list[str] = Field(default_factory=list)
    source_url: Optional[HttpUrl] = None
    version: Optional[str] = None
    page_count: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    page_start: int
    page_end: int
    token_count: int
    metadata: LegalMetadata


class IngestedDocument(BaseModel):
    document_id: str
    raw_text: str
    pages: list[str]
    metadata: LegalMetadata
    chunks: list[DocumentChunk]
    supersession_relations: list[SupersessionRelation] = Field(default_factory=list)


class MetadataFilter(BaseModel):
    document_type: Optional[DocumentType] = None
    department: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    issuing_authority: Optional[str] = None
    source_document: Optional[str] = None


class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    score: float


class Citation(BaseModel):
    document_id: str
    title: str
    circular_number: Optional[str] = None
    notification_number: Optional[str] = None
    publication_date: Optional[date] = None
    page_start: int
    page_end: int
    source_url: Optional[str] = None


class AnswerResponse(BaseModel):
    answer_markdown: str
    citations: list[Citation]
    supersession_warnings: list[str] = Field(default_factory=list)


class CompareResponse(BaseModel):
    comparison_markdown: str
    citations: list[Citation]
    supersession_warnings: list[str] = Field(default_factory=list)
