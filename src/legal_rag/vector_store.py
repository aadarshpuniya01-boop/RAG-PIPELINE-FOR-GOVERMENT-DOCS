from typing import Iterable
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models

from legal_rag.config import get_settings
from legal_rag.schemas import DocumentChunk, MetadataFilter


def get_client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url)


def ensure_collection(client: QdrantClient, vector_size: int) -> None:
    settings = get_settings()
    collections = client.get_collections().collections
    if any(collection.name == settings.qdrant_collection for collection in collections):
        return
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20000),
    )
    for field in ["document_type", "department", "issuing_authority", "document_id"]:
        client.create_payload_index(settings.qdrant_collection, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(
        settings.qdrant_collection,
        field_name="publication_date",
        field_schema=models.PayloadSchemaType.DATETIME,
    )


def chunk_payload(chunk: DocumentChunk) -> dict:
    meta = chunk.metadata
    return {
        "chunk": chunk.model_dump(mode="json"),
        "document_id": chunk.document_id,
        "document_type": meta.document_type.value,
        "department": meta.department,
        "issuing_authority": meta.issuing_authority,
        "publication_date": meta.publication_date.isoformat() if meta.publication_date else None,
        "source_url": str(meta.source_url) if meta.source_url else None,
        "title": meta.title,
        "text": chunk.text,
    }


def upsert_chunks(client: QdrantClient, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
    if not chunks:
        return
    ensure_collection(client, len(vectors[0]))
    points = [
        models.PointStruct(id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)), vector=vector, payload=chunk_payload(chunk))
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    client.upsert(collection_name=get_settings().qdrant_collection, points=points, wait=True)


def build_filter(metadata_filter: MetadataFilter | None) -> models.Filter | None:
    if not metadata_filter:
        return None
    must: list[models.Condition] = []
    if metadata_filter.document_type:
        must.append(models.FieldCondition(key="document_type", match=models.MatchValue(value=metadata_filter.document_type.value)))
    if metadata_filter.department:
        must.append(models.FieldCondition(key="department", match=models.MatchText(text=metadata_filter.department)))
    if metadata_filter.issuing_authority:
        must.append(models.FieldCondition(key="issuing_authority", match=models.MatchText(text=metadata_filter.issuing_authority)))
    if metadata_filter.source_document:
        must.append(models.FieldCondition(key="document_id", match=models.MatchValue(value=metadata_filter.source_document)))
    if metadata_filter.date_from or metadata_filter.date_to:
        must.append(
            models.FieldCondition(
                key="publication_date",
                range=models.DatetimeRange(
                    gte=metadata_filter.date_from.isoformat() if metadata_filter.date_from else None,
                    lte=metadata_filter.date_to.isoformat() if metadata_filter.date_to else None,
                ),
            )
        )
    return models.Filter(must=must) if must else None


def search(client: QdrantClient, query_vector: list[float], metadata_filter: MetadataFilter | None, top_k: int) -> Iterable[models.ScoredPoint]:
    return client.search(
        collection_name=get_settings().qdrant_collection,
        query_vector=query_vector,
        query_filter=build_filter(metadata_filter),
        limit=top_k,
        with_payload=True,
    )
