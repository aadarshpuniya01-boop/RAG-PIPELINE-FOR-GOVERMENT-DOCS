from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "local"
    log_level: str = "INFO"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "government_legal_documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_tokens: int = Field(default=800, ge=500, le=1000)
    chunk_overlap_tokens: int = Field(default=120, ge=0, le=250)
    retrieval_top_k: int = Field(default=12, ge=10, le=20)
    rerank_top_n: int = Field(default=8, ge=1, le=20)
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    tesseract_cmd: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
