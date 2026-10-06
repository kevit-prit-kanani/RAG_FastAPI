from __future__ import annotations

from dataclasses import dataclass
import asyncio
from typing import Protocol

from app.models.schemas import RetrievedChunk, SearchFilters, SearchRequest

RRF_K = 60


@dataclass(slots=True)
class SearchHit:
    chunk_id: str
    page_number: int
    section_path: list[str]
    content: str
    score: float


@dataclass(slots=True)
class RankedList:
    modality: str
    hits: list[SearchHit]


class RetrievalRepository(Protocol):
    async def vector_search(
        self,
        query_vector: list[float],
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]: ...

    async def keyword_search(
        self,
        query: str,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]: ...


def reciprocal_rank_fusion(rankings: list[RankedList], rrf_k: int = RRF_K) -> list[RetrievedChunk]:
    """Fuse multiple ranked lists without comparing incompatible raw score scales."""
    scores: dict[str, float] = {}
    best: dict[str, SearchHit] = {}
    best_vector_score: dict[str, float] = {}
    best_keyword_score: dict[str, float] = {}

    for ranked_list in rankings:
        for rank, hit in enumerate(ranked_list.hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            best.setdefault(hit.chunk_id, hit)
            if ranked_list.modality == "vector":
                best_vector_score[hit.chunk_id] = max(best_vector_score.get(hit.chunk_id, float("-inf")), hit.score)
            else:
                best_keyword_score[hit.chunk_id] = max(best_keyword_score.get(hit.chunk_id, float("-inf")), hit.score)

    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            page_number=best[chunk_id].page_number,
            section_path=best[chunk_id].section_path,
            content=best[chunk_id].content,
            fused_score=scores[chunk_id],
            vector_score=best_vector_score.get(chunk_id),
            keyword_score=best_keyword_score.get(chunk_id),
        )
        for chunk_id in ordered
    ]


class HybridRetriever:
    def __init__(self, repository: RetrievalRepository, embedding_service) -> None:
        self.repository = repository
        self.embedding_service = embedding_service

    async def search(self, request: SearchRequest) -> list[RetrievedChunk]:
        queries = _dedupe_queries([request.query, *request.alternate_queries])
        per_query_limit = max(request.top_k * 2, 10)

        # Batch query embeddings into one OpenAI request rather than one request per reformulation.
        vectors = await self.embedding_service.embed_queries(queries)
        rankings: list[RankedList] = []

        async def run_query(query: str, vector: list[float]) -> tuple[list[SearchHit], list[SearchHit]]:
            return await asyncio.gather(
                self.repository.vector_search(vector, per_query_limit, request.filters),
                self.repository.keyword_search(query, per_query_limit, request.filters),
            )

        per_query_results = await asyncio.gather(
            *(run_query(query, vector) for query, vector in zip(queries, vectors, strict=True))
        )
        for vector_hits, keyword_hits in per_query_results:
            rankings.append(RankedList("vector", vector_hits))
            rankings.append(RankedList("keyword", keyword_hits))

        if request.keywords:
            keyword_query = " ".join(_dedupe_strings(request.keywords))
            keyword_hits = await self.repository.keyword_search(keyword_query, per_query_limit, request.filters)
            rankings.append(RankedList("keyword", keyword_hits))

        fused = reciprocal_rank_fusion(rankings)
        return fused[: request.top_k]


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = " ".join(value.split()).strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            output.append(normalized)
    return output


def _dedupe_queries(values: list[str]) -> list[str]:
    return _dedupe_strings(values)[:4]
