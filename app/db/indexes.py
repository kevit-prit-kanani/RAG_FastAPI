from __future__ import annotations

import asyncio
import time

from pymongo.operations import SearchIndexModel

from app.core.config import Settings
from app.db.mongodb import MongoDB


def _index_models(settings: Settings) -> list[SearchIndexModel]:
    text_index = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": False,
                "fields": {
                    "content": {"type": "string"},
                    "section_text": {"type": "string"},
                    "source_filename": {"type": "string"},
                },
            }
        },
        name=settings.mongodb_search_index,
        type="search",
    )

    vector_index = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": settings.openai_embedding_dimensions,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "document_id"},
                {"type": "filter", "path": "page_number"},
            ]
        },
        name=settings.mongodb_vector_index,
        type="vectorSearch",
    )
    return [text_index, vector_index]


async def list_search_indexes(mongo: MongoDB) -> list[dict]:
    """Return Atlas Search/Vector Search indexes, including their build status."""
    cursor = await mongo.chunks.list_search_indexes()
    return await cursor.to_list(length=None)


async def create_search_indexes(mongo: MongoDB, settings: Settings) -> list[str]:
    """Create only missing Atlas Search/Vector Search indexes."""
    existing = await list_search_indexes(mongo)
    existing_names = {str(item.get("name")) for item in existing}
    models = [model for model in _index_models(settings) if model.document["name"] not in existing_names]
    if not models:
        return []
    return await mongo.chunks.create_search_indexes(models=models)


async def wait_for_search_indexes(
    mongo: MongoDB,
    settings: Settings,
    timeout_seconds: float = 60.0,
    poll_seconds: float = 2.0,
) -> list[dict]:
    """Wait until the configured search indexes are READY or fail clearly."""
    wanted = {settings.mongodb_search_index, settings.mongodb_vector_index}
    deadline = time.monotonic() + timeout_seconds
    latest: list[dict] = []

    while True:
        latest = await list_search_indexes(mongo)
        by_name = {str(item.get("name")): item for item in latest}
        missing = sorted(wanted - set(by_name))
        if not missing:
            failed = {
                name: by_name[name].get("status")
                for name in wanted
                if by_name[name].get("status") in {"FAILED", "FAILURE"}
            }
            if failed:
                raise RuntimeError(f"Atlas Search index build failed: {failed}")
            statuses = {name: by_name[name].get("status") for name in sorted(wanted)}
            if all(status == "READY" for status in statuses.values()):
                return [by_name[name] for name in sorted(wanted)]
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for Atlas Search indexes. Current indexes: {latest}")
        await asyncio.sleep(poll_seconds)
