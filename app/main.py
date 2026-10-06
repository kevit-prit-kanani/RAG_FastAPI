from fastapi import FastAPI

from app.api.ingest import router as ingest_router
from app.api.search import router as search_router
from app.core.config import get_settings
from app.api.chat import router as chat_router

settings = get_settings()
app = FastAPI(title="FastAPI MongoDB RAG Prototype", version="0.6.0")
app.include_router(ingest_router)
app.include_router(search_router)
app.include_router(chat_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
