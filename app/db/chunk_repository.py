from __future__ import annotations

from datetime import datetime, timezone

from pymongo import ReplaceOne

from app.core.config import Settings
from app.db.mongodb import MongoDB
from app.db.search_pipelines import build_keyword_pipeline, build_vector_pipeline
from app.models.schemas import EmbeddedChunk, SearchFilters
from app.services.retrieval import SearchHit


class ChunkRepository:
    def __init__(self, mongo: MongoDB, settings: Settings) -> None:
        self.settings = settings
        self.collection = mongo.chunks

    async def ensure_indexes(self) -> None:
        await self.collection.create_index("chunk_id", unique=True)
        await self.collection.create_index("document_id")

    async def upsert_chunks(self, chunks: list[EmbeddedChunk]) -> int:
        if not chunks:
            return 0
        now = datetime.now(timezone.utc)
        operations = []
        for chunk in chunks:
            document = chunk.model_dump(mode="json")
            document["_id"] = chunk.chunk_id
            document["section_text"] = " > ".join(chunk.section_path)
            document["created_at"] = now
            document["updated_at"] = now
            operations.append(ReplaceOne({"_id": chunk.chunk_id}, document, upsert=True))
        await self.collection.bulk_write(operations, ordered=False)
        return len(chunks)

    async def vector_search(
        self,
        query_vector: list[float],
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        pipeline = build_vector_pipeline(self.settings, query_vector, limit, filters)
        cursor = await self.collection.aggregate(pipeline)
        rows = await cursor.to_list(length=limit)
        rows = self._apply_post_filters(rows, filters)
        return [self._to_hit(row) for row in rows]

    async def keyword_search(
        self,
        query: str,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        if not query.strip():
            return []
        pipeline = build_keyword_pipeline(self.settings, query, limit)
        cursor = await self.collection.aggregate(pipeline)
        rows = await cursor.to_list(length=limit)
        rows = self._apply_post_filters(rows, filters)
        return [self._to_hit(row) for row in rows]

    @staticmethod
    def _apply_post_filters(rows: list[dict], filters: SearchFilters | None) -> list[dict]:
        if filters is None:
            return rows
        output = rows
        if filters.section:
            needle = filters.section.casefold()
            output = [
                row for row in output
                if needle in " > ".join(row.get("section_path", [])).casefold()
            ]
        return output

    @staticmethod
    def _to_hit(row: dict) -> SearchHit:
        return SearchHit(
            chunk_id=str(row["chunk_id"]),
            page_number=int(row["page_number"]),
            section_path=list(row.get("section_path", [])),
            content=str(row["content"]),
            score=float(row.get("score", 0.0)),
        )
