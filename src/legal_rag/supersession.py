import re

from legal_rag.schemas import DocumentChunk, SupersessionRelation, SupersessionType


PATTERNS: list[tuple[SupersessionType, re.Pattern[str]]] = [
    (SupersessionType.SUPERSEDES, re.compile(r"(?:supersedes|in\s+supersession\s+of)\s+([^.\n;]+)", re.I)),
    (SupersessionType.PARTIAL_MODIFICATION, re.compile(r"in\s+partial\s+modification\s+of\s+([^.\n;]+)", re.I)),
    (SupersessionType.AMENDS, re.compile(r"amend(?:ment|s|ed)?\s+(?:to|of)?\s*([^.\n;]+)", re.I)),
    (SupersessionType.REPLACES, re.compile(r"(?:shall\s+replace|replaces?)\s+([^.\n;]+)", re.I)),
]


def detect_supersession(text: str) -> list[SupersessionRelation]:
    relations: list[SupersessionRelation] = []
    for relation_type, pattern in PATTERNS:
        for match in pattern.finditer(text):
            target = match.group(1).strip(" :-")
            evidence = match.group(0).strip()
            confidence = 0.92 if relation_type in {SupersessionType.SUPERSEDES, SupersessionType.REPLACES} else 0.82
            relations.append(
                SupersessionRelation(
                    relation_type=relation_type,
                    target_reference=target[:240],
                    evidence_text=evidence[:500],
                    confidence=confidence,
                )
            )
    return relations


def build_supersession_warnings(retrieved_chunks: list[DocumentChunk], newer_relations: list[SupersessionRelation]) -> list[str]:
    warnings: list[str] = []
    seen: set[str] = set()
    for chunk in retrieved_chunks:
        identifiers = [chunk.metadata.circular_number, chunk.metadata.notification_number, chunk.metadata.title]
        for relation in newer_relations:
            if any(identifier and identifier.lower() in relation.target_reference.lower() for identifier in identifiers):
                key = f"{chunk.document_id}:{relation.target_reference}"
                if key in seen:
                    continue
                seen.add(key)
                warnings.append(
                    "⚠ Supersession Warning: "
                    f"{identifiers[0] or identifiers[1] or chunk.metadata.title} may be superseded or modified by "
                    f"a newer document referencing '{relation.target_reference}'. Evidence: {relation.evidence_text}"
                )
    return warnings
