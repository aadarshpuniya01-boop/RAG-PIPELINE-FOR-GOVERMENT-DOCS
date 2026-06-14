from pathlib import Path

from legal_rag.chunker import chunk_document
from legal_rag.config import get_settings
from legal_rag.embeddings import embed_texts
from legal_rag.metadata_extractor import extract_metadata
from legal_rag.pdf_loader import extract_pdf_text
from legal_rag.schemas import IngestedDocument
from legal_rag.supersession import detect_supersession
from legal_rag.vector_store import get_client, upsert_chunks


def ingest_pdf(path: str | Path, source_url: str | None = None, persist: bool = True) -> IngestedDocument:
    settings = get_settings()
    pdf_path = Path(path)
    pages = extract_pdf_text(pdf_path)
    raw_text = "\n\n".join(pages)
    metadata = extract_metadata(raw_text, pdf_path.name, page_count=len(pages), source_url=source_url)
    chunks = chunk_document(pages, metadata, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
    relations = detect_supersession(raw_text)

    if persist and chunks:
        vectors = embed_texts([chunk.text for chunk in chunks])
        upsert_chunks(get_client(), chunks, vectors)

    return IngestedDocument(
        document_id=metadata.document_id,
        raw_text=raw_text,
        pages=pages,
        metadata=metadata,
        chunks=chunks,
        supersession_relations=relations,
    )
