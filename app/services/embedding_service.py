from typing import Protocol

from app.core.config import Settings
from app.models.schemas import Chunk, EmbeddedChunk

EMBED_BATCH_SIZE = 64


class EmbeddingClient(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingService:
    def __init__(self, client: EmbeddingClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    async def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        embedded: list[EmbeddedChunk] = []
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start:start + EMBED_BATCH_SIZE]
            vectors = await self.client.embed([chunk.embedding_text for chunk in batch])
            if len(vectors) != len(batch):
                raise RuntimeError(f"Embedding count mismatch: expected {len(batch)}, got {len(vectors)}")
            embedded.extend(
                EmbeddedChunk(
                    **chunk.model_dump(),
                    embedding=vector,
                    embedding_model=self.settings.openai_embedding_model,
                    embedding_dimensions=self.settings.openai_embedding_dimensions,
                )
                for chunk, vector in zip(batch, vectors, strict=True)
            )
        return embedded

    async def embed_query(self, query: str) -> list[float]:
        vectors = await self.client.embed([query])
        if len(vectors) != 1:
            raise RuntimeError(f"Query embedding count mismatch: expected 1, got {len(vectors)}")
        return vectors[0]

    async def embed_queries(self, queries: list[str]) -> list[list[float]]:
        if not queries:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(queries), EMBED_BATCH_SIZE):
            batch = queries[start:start + EMBED_BATCH_SIZE]
            batch_vectors = await self.client.embed(batch)
            if len(batch_vectors) != len(batch):
                raise RuntimeError(
                    f"Query embedding count mismatch: expected {len(batch)}, got {len(batch_vectors)}"
                )
            vectors.extend(batch_vectors)
        return vectors
