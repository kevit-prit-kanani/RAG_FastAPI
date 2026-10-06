from uuid import uuid4
from unittest.mock import AsyncMock

import asyncio

from app.core.config import Settings
from app.models.schemas import Chunk
from app.services.embedding_service import EmbeddingService


def test_embedding_service_maps_vectors_to_chunks():
    settings = Settings(openai_api_key="x", openai_embedding_dimensions=3)
    client = AsyncMock()
    client.embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    service = EmbeddingService(client, settings)
    chunks = [
        Chunk(chunk_id="1", document_id=uuid4(), source_filename="a.pdf", page_number=1, block_type="paragraph", content="a", embedding_text="a", token_count=1),
        Chunk(chunk_id="2", document_id=uuid4(), source_filename="a.pdf", page_number=1, block_type="paragraph", content="b", embedding_text="b", token_count=1),
    ]
    result = asyncio.run(service.embed_chunks(chunks))
    assert [x.embedding for x in result] == client.embed.return_value
    assert all(x.embedding_dimensions == 3 for x in result)
