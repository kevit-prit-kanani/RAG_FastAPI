from hashlib import sha256
from io import BytesIO
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.db.chunk_repository import ChunkRepository
from app.db.mongodb import MongoDB
from app.llm.openai_client import OpenAIClient
from app.models.schemas import IngestPreviewResponse, IngestResponse
from app.services.chunking import chunk_pages
from app.services.embedding_service import EmbeddingService
from app.services.pdf_loader import extract_pages

router = APIRouter(prefix="/ingest", tags=["ingestion"])


def _document_id(raw: bytes) -> UUID:
    digest = sha256(raw).digest()
    return UUID(bytes=digest[:16], version=4)


@router.post("/preview", response_model=IngestPreviewResponse)
async def preview_ingestion(file: UploadFile = File(...)) -> IngestPreviewResponse:
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Upload a PDF file.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")
    document_id = _document_id(raw)
    try:
        pages = await run_in_threadpool(extract_pages, BytesIO(raw))
        chunks = await run_in_threadpool(chunk_pages, pages, document_id, file.filename or "unknown.pdf")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process PDF: {exc}") from exc
    native = sum(p.extraction_method == "native" for p in pages)
    return IngestPreviewResponse(
        document_id=document_id,
        source_filename=file.filename or "unknown.pdf",
        pages=len(pages),
        pages_with_native_text=native,
        pages_with_ocr=len(pages) - native,
        chunks=len(chunks),
        sample_chunks=chunks[:10],
    )


@router.post("", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Upload a PDF file.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    settings = get_settings()
    source_filename = file.filename or "unknown.pdf"
    document_id = _document_id(raw)

    try:
        pages = await run_in_threadpool(extract_pages, BytesIO(raw))
        chunks = await run_in_threadpool(chunk_pages, pages, document_id, source_filename)
        openai = OpenAIClient(settings)
        embedder = EmbeddingService(openai, settings)
        embedded_chunks = await embedder.embed_chunks(chunks)
        mongo = MongoDB(settings)
        try:
            await mongo.ping()
            repository = ChunkRepository(mongo, settings)
            await repository.ensure_indexes()
            stored = await repository.upsert_chunks(embedded_chunks)
        finally:
            await mongo.close()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ingestion failed: {exc}") from exc

    return IngestResponse(
        document_id=document_id,
        source_filename=source_filename,
        pages=len(pages),
        chunks=len(chunks),
        embedded=len(embedded_chunks),
        stored=stored,
        embedding_model=settings.openai_embedding_model,
        embedding_dimensions=settings.openai_embedding_dimensions,
    )
