try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # Allows lightweight unit tests before full requirements are installed.
    RecursiveCharacterTextSplitter = None

from legal_rag.schemas import DocumentChunk, LegalMetadata


def approx_token_count(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def chunk_document(pages: list[str], metadata: LegalMetadata, chunk_size_tokens: int, overlap_tokens: int) -> list[DocumentChunk]:
    chunk_size_chars = chunk_size_tokens * 4
    overlap_chars = overlap_tokens * 4
    chunks: list[DocumentChunk] = []
    page_offsets: list[tuple[int, int, int]] = []
    cursor = 0
    full_text = ""
    for index, page in enumerate(pages, start=1):
        start = cursor
        full_text += page + "\n\n"
        cursor = len(full_text)
        page_offsets.append((index, start, cursor))

    chunk_texts = _split_text(full_text, chunk_size_chars, overlap_chars)
    for idx, chunk_text in enumerate(chunk_texts, start=1):
        start = full_text.find(chunk_text[:80])
        end = start + len(chunk_text) if start >= 0 else start
        touched = [page_no for page_no, p_start, p_end in page_offsets if start < p_end and end > p_start]
        page_start = min(touched) if touched else 1
        page_end = max(touched) if touched else metadata.page_count
        chunks.append(
            DocumentChunk(
                chunk_id=f"{metadata.document_id}:{idx:04d}",
                document_id=metadata.document_id,
                text=chunk_text.strip(),
                page_start=page_start,
                page_end=page_end,
                token_count=approx_token_count(chunk_text),
                metadata=metadata,
            )
        )
    return chunks


def _split_text(text: str, chunk_size_chars: int, overlap_chars: int) -> list[str]:
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size_chars,
            chunk_overlap=overlap_chars,
            separators=["\n\n", "\n", ". ", "; ", " ", ""],
        )
        return splitter.split_text(text)

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size_chars)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return chunks
