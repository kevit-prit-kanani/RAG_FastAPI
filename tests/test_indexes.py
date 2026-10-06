from app.core.config import Settings
from app.db.search_pipelines import build_vector_pipeline


def test_vector_pipeline_uses_configured_index_and_embedding_path():
    settings = Settings(
        mongodb_uri="mongodb://example",
        mongodb_vector_index="chunk_vector_index",
        openai_embedding_dimensions=3072,
    )
    pipeline = build_vector_pipeline(settings, [0.0] * 3072, 8)
    stage = pipeline[0]["$vectorSearch"]
    assert stage["index"] == "chunk_vector_index"
    assert stage["path"] == "embedding"
    assert stage["limit"] == 8


import asyncio
from unittest.mock import AsyncMock, Mock


def test_list_search_indexes_awaits_async_collection_method():
    from app.db.indexes import list_search_indexes

    cursor = Mock()
    cursor.to_list = AsyncMock(return_value=[{"name": "chunk_vector_index", "status": "READY"}])
    mongo = Mock()
    mongo.chunks.list_search_indexes = AsyncMock(return_value=cursor)

    result = asyncio.run(list_search_indexes(mongo))

    assert result == [{"name": "chunk_vector_index", "status": "READY"}]
    mongo.chunks.list_search_indexes.assert_awaited_once()
    cursor.to_list.assert_awaited_once_with(length=None)
