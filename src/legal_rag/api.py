import logging
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from legal_rag.comparison import compare_documents
from legal_rag.config import get_settings
from legal_rag.logging_config import configure_logging
from legal_rag.pipeline import ingest_pdf
from legal_rag.retriever import answer_question
from legal_rag.schemas import AnswerResponse, CompareResponse, IngestedDocument, MetadataFilter

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Government Legal RAG API",
    version="0.1.0",
    description="RAG pipeline for circulars, notifications, gazettes, office memoranda, amendments, orders, and guidelines.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}


@app.post("/ingest", response_model=IngestedDocument)
async def ingest_endpoint(
    file: Annotated[UploadFile, File()],
    source_url: Annotated[str | None, Form()] = None,
) -> IngestedDocument:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / file.filename
        pdf_path.write_bytes(await file.read())
        try:
            return ingest_pdf(pdf_path, source_url=source_url, persist=True)
        except Exception as exc:
            logger.exception("Ingestion failed")
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@app.post("/query", response_model=AnswerResponse)
async def query_endpoint(question: str, metadata_filter: MetadataFilter | None = None) -> AnswerResponse:
    try:
        return await answer_question(question, metadata_filter)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


@app.post("/compare", response_model=CompareResponse)
async def compare_endpoint(document_a: str, document_b: str, question: str = "Compare both documents.") -> CompareResponse:
    try:
        return await compare_documents(document_a, document_b, question)
    except Exception as exc:
        logger.exception("Comparison failed")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {exc}") from exc
