try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from legal_rag.citations import citation_from_chunk, format_citation
from legal_rag.config import get_settings
from legal_rag.embeddings import embed_texts
from legal_rag.llm import LLMClient
from legal_rag.prompts import ANSWER_PROMPT
from legal_rag.schemas import AnswerResponse, DocumentChunk, MetadataFilter, RetrievedChunk
from legal_rag.supersession import SupersessionRelation, build_supersession_warnings
from legal_rag.vector_store import get_client, search


def _point_to_chunk(point) -> DocumentChunk:
    return DocumentChunk.model_validate(point.payload["chunk"])


def _bm25_rerank(query: str, retrieved: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
    if len(retrieved) <= 1:
        return retrieved
    if BM25Okapi is None:
        return sorted(retrieved, key=lambda item: item.score, reverse=True)[:top_n]
    corpus = [item.chunk.text.lower().split() for item in retrieved]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.lower().split())
    rescored: list[RetrievedChunk] = []
    max_bm25 = max(scores) if len(scores) else 0
    for item, bm25_score in zip(retrieved, scores, strict=True):
        normalized_bm25 = float(bm25_score / max_bm25) if max_bm25 else 0.0
        hybrid_score = 0.75 * item.score + 0.25 * normalized_bm25
        rescored.append(RetrievedChunk(chunk=item.chunk, score=hybrid_score))
    return sorted(rescored, key=lambda item: item.score, reverse=True)[:top_n]


def retrieve_chunks(query: str, metadata_filter: MetadataFilter | None = None, top_k: int | None = None) -> list[RetrievedChunk]:
    settings = get_settings()
    vector = embed_texts([query])[0]
    points = search(get_client(), vector, metadata_filter, top_k or settings.retrieval_top_k)
    dense_hits = [RetrievedChunk(chunk=_point_to_chunk(point), score=float(point.score or 0.0)) for point in points]
    return _bm25_rerank(query, dense_hits, settings.rerank_top_n)


def context_block(chunks: list[DocumentChunk]) -> str:
    lines: list[str] = []
    for chunk in chunks:
        citation = format_citation(citation_from_chunk(chunk))
        lines.append(f"[{citation}] {chunk.text}")
    return "\n\n".join(lines)


async def answer_question(
    question: str,
    metadata_filter: MetadataFilter | None = None,
    known_supersession_relations: list[SupersessionRelation] | None = None,
) -> AnswerResponse:
    retrieved = retrieve_chunks(question, metadata_filter)
    chunks = [item.chunk for item in retrieved]
    warnings = build_supersession_warnings(chunks, known_supersession_relations or [])
    prompt = ANSWER_PROMPT.format(
        question=question,
        context=context_block(chunks),
        warnings="\n".join(warnings) if warnings else "None",
    )
    answer = await LLMClient().complete(prompt)
    citations = []
    seen = set()
    for chunk in chunks:
        citation = citation_from_chunk(chunk)
        key = citation.model_dump_json()
        if key not in seen:
            seen.add(key)
            citations.append(citation)
    return AnswerResponse(answer_markdown=answer, citations=citations, supersession_warnings=warnings)
