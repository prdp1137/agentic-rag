"""
LLM Configuration and Factory.

Centralizes LLM instantiation with consistent settings.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from src.config.settings import get_settings


@lru_cache
def get_llm() -> ChatOpenAI:
    """
    Get the primary LLM for reasoning and generation.
    
    Uses LRU cache to reuse the same LLM instance across calls,
    reducing overhead and ensuring consistent configuration.
    
    Returns:
        Configured ChatOpenAI instance.
    """
    settings = get_settings()
    
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
