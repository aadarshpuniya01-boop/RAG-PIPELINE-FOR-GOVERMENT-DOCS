from collections import defaultdict

from legal_rag.citations import citation_from_chunk
from legal_rag.llm import LLMClient
from legal_rag.prompts import COMPARISON_PROMPT
from legal_rag.retriever import context_block, retrieve_chunks
from legal_rag.schemas import CompareResponse, MetadataFilter, SupersessionRelation
from legal_rag.supersession import build_supersession_warnings


async def compare_documents(
    document_a: str,
    document_b: str,
    question: str = "Compare legal and procedural differences.",
    known_supersession_relations: list[SupersessionRelation] | None = None,
) -> CompareResponse:
    chunks_a = [hit.chunk for hit in retrieve_chunks(question, MetadataFilter(source_document=document_a))]
    chunks_b = [hit.chunk for hit in retrieve_chunks(question, MetadataFilter(source_document=document_b))]
    prompt = COMPARISON_PROMPT.format(document_a=context_block(chunks_a), document_b=context_block(chunks_b))
    markdown = await LLMClient().complete(prompt)
    warnings = build_supersession_warnings(chunks_a + chunks_b, known_supersession_relations or [])
    citations = []
    seen = set()
    for chunk in chunks_a + chunks_b:
        citation = citation_from_chunk(chunk)
        key = citation.model_dump_json()
        if key not in seen:
            seen.add(key)
            citations.append(citation)
    if not markdown.startswith("|"):
        markdown = _deterministic_comparison_table(document_a, document_b, chunks_a, chunks_b)
    return CompareResponse(comparison_markdown=markdown, citations=citations, supersession_warnings=warnings)


def _deterministic_comparison_table(document_a: str, document_b: str, chunks_a, chunks_b) -> str:
    buckets: dict[str, list[str]] = defaultdict(list)
    keywords = {
        "dates": ["date", "effective", "published", "force"],
        "eligibility": ["eligible", "eligibility", "applicant", "beneficiary"],
        "procedure": ["procedure", "submit", "form", "approval", "authority"],
        "policy": ["shall", "must", "may", "policy", "scheme"],
    }
    for label, words in keywords.items():
        for chunk in chunks_a:
            if any(word in chunk.text.lower() for word in words):
                buckets[f"a_{label}"].append(chunk.text[:240])
        for chunk in chunks_b:
            if any(word in chunk.text.lower() for word in words):
                buckets[f"b_{label}"].append(chunk.text[:240])

    rows = ["| Aspect | Document A | Document B | Difference | Legal effect |", "|---|---|---|---|---|"]
    for label in ["dates", "eligibility", "procedure", "policy"]:
        a_text = " ".join(buckets.get(f"a_{label}", []))[:350] or "No strong evidence retrieved."
        b_text = " ".join(buckets.get(f"b_{label}", []))[:350] or "No strong evidence retrieved."
        rows.append(f"| {label.title()} | {a_text} | {b_text} | Review highlighted evidence. | Requires legal validation. |")
    return "\n".join(rows)
