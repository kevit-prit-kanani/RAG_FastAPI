from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.db.chunk_repository import ChunkRepository
from app.db.mongodb import MongoDB
from app.llm.openai_client import OpenAIClient
from app.models.schemas import ChatRequest, ChatResponse
from app.services.embedding_service import EmbeddingService
from app.services.retrieval import HybridRetriever
from app.services.conversation import ConversationRepository
from app.services.rag import RAGService

router = APIRouter(prefix="/chat", tags=["generation"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    settings = get_settings()
    mongo = MongoDB(settings)
    try:
        await mongo.ping()
        repository = ChunkRepository(mongo, settings)
        openai = OpenAIClient(settings)
        embedder = EmbeddingService(openai, settings)
        retriever = HybridRetriever(repository, embedder)
        conversations = ConversationRepository(mongo, settings)
        service = RAGService(openai, retriever, conversations)
        answer, conversation_id, _retrieved = await service.answer(request)
        return ChatResponse(conversation_id=conversation_id, **answer.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"RAG request failed: {exc}") from exc
    finally:
        await mongo.close()
