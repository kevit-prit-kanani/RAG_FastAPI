from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.db.chunk_repository import ChunkRepository
from app.db.mongodb import MongoDB
from app.llm.openai_client import OpenAIClient
from app.models.schemas import SearchRequest, SearchResponse
from app.services.embedding_service import EmbeddingService
from app.services.retrieval import HybridRetriever

router = APIRouter(prefix="/search", tags=["retrieval"])


@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    settings = get_settings()
    try:
        openai = OpenAIClient(settings)
        embedder = EmbeddingService(openai, settings)
        mongo = MongoDB(settings)
        try:
            await mongo.ping()
            repository = ChunkRepository(mongo, settings)
            retriever = HybridRetriever(repository, embedder)
            results = await retriever.search(request)
        finally:
            await mongo.close()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Search failed: {exc}") from exc
    queries = [request.query, *request.alternate_queries]
    return SearchResponse(query=request.query, queries_used=queries[:4], results=results)
