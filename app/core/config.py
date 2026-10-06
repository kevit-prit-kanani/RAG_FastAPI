from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True)

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_embedding_model: str = Field(default="text-embedding-3-large", alias="OPENAI_EMBEDDING_MODEL")
    openai_embedding_dimensions: int = Field(default=3072, ge=1, le=8192, alias="OPENAI_EMBEDDING_DIMENSIONS")
    openai_chat_model: str = Field(default="gpt-6.1-sol", alias="OPENAI_CHAT_MODEL")

    mongodb_uri: str | None = Field(default=None, alias="MONGODB_URI")
    mongodb_db: str = Field(default="rag_demo", alias="MONGODB_DB")
    mongodb_chunks_collection: str = Field(default="chunks", alias="MONGODB_CHUNKS_COLLECTION")
    mongodb_conversations_collection: str = Field(default="conversations", alias="MONGODB_CONVERSATIONS_COLLECTION")
    mongodb_vector_index: str = Field(default="chunk_vector_index", alias="MONGODB_VECTOR_INDEX")
    mongodb_search_index: str = Field(default="chunk_text_index", alias="MONGODB_SEARCH_INDEX")

    def require_openai(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        return self.openai_api_key

    def require_mongodb(self) -> str:
        if not self.mongodb_uri:
            raise RuntimeError("MONGODB_URI is not configured")
        return self.mongodb_uri


@lru_cache
def get_settings() -> Settings:
    return Settings()
