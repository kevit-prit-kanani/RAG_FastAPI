import asyncio

from app.models.schemas import SearchRequest
from app.services.retrieval import HybridRetriever, RankedList, SearchHit, reciprocal_rank_fusion


class FakeRepo:
    async def vector_search(self, query_vector, limit, filters=None):
        return [
            SearchHit("a", 1, ["Overview"], "semantic A", 0.90),
            SearchHit("b", 2, ["Faculty"], "semantic B", 0.80),
        ]

    async def keyword_search(self, query, limit, filters=None):
        return [
            SearchHit("b", 2, ["Faculty"], "keyword B", 7.0),
            SearchHit("c", 3, ["Placement"], "keyword C", 5.0),
        ]


class FakeEmbedder:
    async def embed_queries(self, queries):
        return [[0.1, 0.2, 0.3] for _ in queries]


def test_rrf_prefers_cross_retriever_overlap():
    a = [SearchHit("a", 1, [], "a", 1.0), SearchHit("b", 1, [], "b", 0.9)]
    b = [SearchHit("b", 1, [], "b", 2.0), SearchHit("c", 1, [], "c", 1.5)]
    result = reciprocal_rank_fusion([RankedList("vector", a), RankedList("keyword", b)])
    assert result[0].chunk_id == "b"
    assert result[0].fused_score > result[1].fused_score


def test_hybrid_search_returns_top_k_and_uses_batch_embedding():
    retriever = HybridRetriever(FakeRepo(), FakeEmbedder())
    result = asyncio.run(
        retriever.search(
            SearchRequest(
                query="faculty",
                alternate_queries=["healthcare faculty"],
                keywords=["faculty", "Healthcare Program"],
                top_k=2,
            )
        )
    )
    assert len(result) == 2
    assert result[0].chunk_id == "b"
    assert result[0].vector_score is not None
    assert result[0].keyword_score is not None
