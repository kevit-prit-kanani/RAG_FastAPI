import asyncio

from app.db.chunk_repository import ChunkRepository


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    async def to_list(self, length):
        return self.rows[:length]


class FakeCollection:
    def __init__(self):
        self.calls = []

    async def aggregate(self, pipeline):
        self.calls.append(pipeline)
        return FakeCursor([
            {
                "chunk_id": "c1",
                "page_number": 7,
                "section_path": ["Faculty"],
                "content": "Dr. Example",
                "score": 1.0,
            }
        ])


class FakeSettings:
    mongodb_vector_index = "vector"
    mongodb_search_index = "search"


class FakeMongo:  # only the chunks collection is required by the repository
    def __init__(self):
        self.chunks = FakeCollection()


def test_async_aggregate_is_awaited_before_to_list():
    repository = ChunkRepository(FakeMongo(), FakeSettings())
    result = asyncio.run(repository.keyword_search("faculty", 1))

    assert len(result) == 1
    assert result[0].chunk_id == "c1"


def test_async_vector_aggregate_is_awaited_before_to_list():
    repository = ChunkRepository(FakeMongo(), FakeSettings())
    result = asyncio.run(repository.vector_search([0.1, 0.2], 1))

    assert len(result) == 1
    assert result[0].page_number == 7
