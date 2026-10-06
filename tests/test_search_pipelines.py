from uuid import UUID

from app.core.config import Settings
from app.db.search_pipelines import build_keyword_pipeline, build_vector_filter, build_vector_pipeline
from app.models.schemas import SearchFilters


def settings() -> Settings:
    return Settings(
        openai_api_key="x",
        mongodb_uri="mongodb://example",
        mongodb_vector_index="vector_idx",
        mongodb_search_index="search_idx",
        openai_embedding_dimensions=3,
    )


def test_vector_pipeline_has_expected_ann_shape():
    pipeline = build_vector_pipeline(settings(), [0.1, 0.2, 0.3], limit=8)
    stage = pipeline[0]["$vectorSearch"]
    assert stage["index"] == "vector_idx"
    assert stage["path"] == "embedding"
    assert stage["limit"] == 8
    assert stage["numCandidates"] == 160
    assert stage["queryVector"] == [0.1, 0.2, 0.3]
    assert pipeline[1]["$project"]["score"]["$meta"] == "vectorSearchScore"


def test_vector_filter_uses_indexable_fields():
    filters = SearchFilters(
        document_id=UUID("12345678-1234-5678-1234-567812345678"),
        pages=[7, 11],
    )
    expected = {
        "$and": [
            {"document_id": "12345678-1234-5678-1234-567812345678"},
            {"page_number": {"$in": [7, 11]}},
        ]
    }
    assert build_vector_filter(filters) == expected


def test_keyword_pipeline_uses_search_score():
    pipeline = build_keyword_pipeline(settings(), "healthcare faculty", 8)
    assert pipeline[0]["$search"]["index"] == "search_idx"
    assert pipeline[1] == {"$limit": 8}
    assert pipeline[2]["$project"]["score"]["$meta"] == "searchScore"
