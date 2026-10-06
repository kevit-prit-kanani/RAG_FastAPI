from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PageText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    text: str
    extraction_method: Literal["native", "ocr"]
    ocr_confidence: float | None = Field(default=None, ge=0, le=100)


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: UUID
    source_filename: str
    page_number: int = Field(ge=1)
    section_path: list[str] = Field(default_factory=list)
    block_type: Literal["heading", "paragraph", "bullet_list", "table", "mixed"]
    content: str = Field(min_length=1)
    embedding_text: str = Field(min_length=1)
    token_count: int = Field(ge=1)


class EmbeddedChunk(Chunk):
    embedding: list[float] = Field(min_length=1)
    embedding_model: str
    embedding_dimensions: int = Field(ge=1)


class IngestPreviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    source_filename: str
    pages: int
    pages_with_native_text: int
    pages_with_ocr: int
    chunks: int
    sample_chunks: list[Chunk]


class IngestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    source_filename: str
    pages: int = Field(ge=0)
    chunks: int = Field(ge=0)
    embedded: int = Field(ge=0)
    stored: int = Field(ge=0)
    embedding_model: str
    embedding_dimensions: int


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=2000)
    alternate_queries: list[str] = Field(default_factory=list, max_length=4)
    keywords: list[str] = Field(default_factory=list, max_length=12)
    top_k: int = Field(default=8, ge=1, le=20)
    filters: "SearchFilters | None" = None


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: UUID | None = None
    pages: list[int] | None = Field(default=None, max_length=50)
    section: str | None = None


class SearchToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: list[str] = Field(min_length=1, max_length=4)
    keywords: list[str] = Field(default_factory=list, max_length=12)
    top_k: int = Field(default=8, ge=1, le=20)
    filters: SearchFilters | None


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    page_number: int
    section_path: list[str] = Field(default_factory=list)
    content: str
    fused_score: float
    vector_score: float | None = None
    keyword_score: float | None = None


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    queries_used: list[str]
    results: list[RetrievedChunk]


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    page_number: int


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=2, max_length=4000)
    conversation_id: UUID | None = None
    top_k: int = Field(default=8, ge=1, le=20)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str
    created_at: datetime | None = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: UUID
    answer: str = Field(min_length=1)
    citations: list[Citation]
    grounded: bool
    insufficient_evidence: bool


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citations: list[Citation]
    grounded: bool
    insufficient_evidence: bool
