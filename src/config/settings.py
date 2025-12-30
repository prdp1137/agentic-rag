"""
Application Settings and Configuration Management.

Centralizes all configuration using Pydantic Settings for
type-safe environment variable handling.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings can be overridden via environment variables or a .env file.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # -------------------------------------------------------------------------
    # OpenAI Configuration
    # -------------------------------------------------------------------------
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for GPT-4o and embeddings",
    )
    
    llm_model: str = Field(
        default="gpt-4.1-mini",
        description="LLM model for reasoning and generation",
    )
    
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature for generation",
    )
    
    llm_max_tokens: int = Field(
        default=4096,
        ge=1,
        description="Maximum tokens for LLM responses",
    )
    
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model",
    )
    
    embedding_dimensions: int = Field(
        default=1536,
        description="Embedding vector dimensions",
    )
    
    # -------------------------------------------------------------------------
    # Qdrant Configuration
    # -------------------------------------------------------------------------
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL",
    )
    
    qdrant_api_key: str | None = Field(
        default=None,
        description="Qdrant Cloud API key (optional for local)",
    )
    
    qdrant_collection_name: str = Field(
        default="enterprise_rag",
        description="Qdrant collection name",
    )
    
    qdrant_timeout: float = Field(
        default=30.0,
        description="Qdrant client timeout in seconds",
    )
    
    # -------------------------------------------------------------------------
    # Retrieval Configuration
    # -------------------------------------------------------------------------
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of documents to retrieve",
    )
    
    retrieval_score_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold for dense search",
    )
    
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="RRF fusion constant",
    )
    
    # -------------------------------------------------------------------------
    # CRAG Configuration
    # -------------------------------------------------------------------------
    max_query_rewrites: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum number of query rewrites before fallback",
    )
    
    # -------------------------------------------------------------------------
    # Persistence Configuration
    # -------------------------------------------------------------------------
    persistence_type: Literal["memory", "postgres"] = Field(
        default="memory",
        description="Persistence backend type",
    )
    
    postgres_connection_string: str | None = Field(
        default=None,
        description="PostgreSQL connection string for persistence",
    )
    
    # -------------------------------------------------------------------------
    # Observability Configuration
    # -------------------------------------------------------------------------
    langchain_tracing_v2: bool = Field(
        default=False,
        description="Enable LangSmith tracing",
    )
    
    langchain_api_key: str | None = Field(
        default=None,
        description="LangSmith API key",
    )
    
    langchain_project: str = Field(
        default="multi-agent-rag",
        description="LangSmith project name",
    )
    
    # -------------------------------------------------------------------------
    # Web Search Configuration
    # -------------------------------------------------------------------------
    tavily_api_key: str | None = Field(
        default=None,
        description="Tavily API key for web search fallback",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Uses LRU cache to ensure settings are loaded only once.
    
    Returns:
        Settings instance with loaded configuration.
    """
    return Settings()
