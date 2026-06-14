from legal_rag.chunker import chunk_document
from legal_rag.schemas import LegalMetadata


def test_chunker_preserves_document_id() -> None:
    metadata = LegalMetadata(document_id="doc1", title="Circular No. 1", page_count=1)
    chunks = chunk_document(["word " * 900], metadata, chunk_size_tokens=500, overlap_tokens=50)
    assert chunks
    assert all(chunk.document_id == "doc1" for chunk in chunks)
    assert all(1 <= chunk.page_start <= chunk.page_end for chunk in chunks)
