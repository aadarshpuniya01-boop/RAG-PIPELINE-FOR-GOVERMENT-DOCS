from legal_rag.schemas import Citation, DocumentChunk


def citation_from_chunk(chunk: DocumentChunk) -> Citation:
    meta = chunk.metadata
    return Citation(
        document_id=chunk.document_id,
        title=meta.title,
        circular_number=meta.circular_number,
        notification_number=meta.notification_number,
        publication_date=meta.publication_date,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        source_url=str(meta.source_url) if meta.source_url else None,
    )


def format_citation(citation: Citation) -> str:
    identifier = citation.circular_number or citation.notification_number or citation.document_id
    date = citation.publication_date.isoformat() if citation.publication_date else "date unavailable"
    pages = f"p. {citation.page_start}" if citation.page_start == citation.page_end else f"pp. {citation.page_start}-{citation.page_end}"
    return f"{identifier}, {date}, {pages}"
