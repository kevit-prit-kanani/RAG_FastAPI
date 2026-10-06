from __future__ import annotations

from app.core.config import Settings
from app.models.schemas import SearchFilters


def build_vector_pipeline(
    settings: Settings,
    query_vector: list[float],
    limit: int,
    filters: SearchFilters | None = None,
) -> list[dict]:
    stage: dict = {
        "index": settings.mongodb_vector_index,
        "path": "embedding",
        "queryVector": query_vector,
        "numCandidates": max(limit * 20, 100),
        "limit": limit,
    }
    vector_filter = build_vector_filter(filters)
    if vector_filter:
        stage["filter"] = vector_filter

    return [
        {"$vectorSearch": stage},
        {
            "$project": {
                "_id": 0,
                "chunk_id": 1,
                "document_id": 1,
                "page_number": 1,
                "section_path": 1,
                "content": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]


def build_vector_filter(filters: SearchFilters | None) -> dict | None:
    if filters is None:
        return None

    clauses: list[dict] = []
    if filters.document_id is not None:
        clauses.append({"document_id": str(filters.document_id)})
    if filters.pages:
        clauses.append({"page_number": {"$in": filters.pages}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def build_keyword_pipeline(settings: Settings, query: str, limit: int) -> list[dict]:
    return [
        {
            "$search": {
                "index": settings.mongodb_search_index,
                "text": {
                    "query": query,
                    "path": ["content", "section_text", "source_filename"],
                    "matchCriteria": "any",
                },
            }
        },
        {"$limit": limit},
        {
            "$project": {
                "_id": 0,
                "chunk_id": 1,
                "document_id": 1,
                "page_number": 1,
                "section_path": 1,
                "content": 1,
                "score": {"$meta": "searchScore"},
            }
        },
    ]
