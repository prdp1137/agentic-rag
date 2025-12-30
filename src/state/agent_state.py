"""
Agent State Definition (The "Blackboard" Pattern).

This module defines the global state that persists across graph execution,
enabling complex multi-agent coordination without tight coupling.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    """
    Global state that persists across graph execution.
    
    This TypedDict serves as the "Blackboard" pattern where all agents
    read from and write to a shared state, enabling complex multi-agent
    coordination without tight coupling.
    
    Attributes:
        question: The user's original input query.
        documents: The retrieved context chunks from vector DB or web.
        generation: The LLM's generated response.
        web_search: Binary flag ('Yes'/'No') indicating if web fallback is needed.
        grading_status: Outcome of relevance check ('relevant'/'irrelevant'/'pending').
        query_rewrite_count: Number of times the query has been rewritten
                            (prevents infinite loops in correction cycle).
    
    Example:
        >>> state: AgentState = {
        ...     "question": "What is LangGraph?",
        ...     "documents": [],
        ...     "generation": "",
        ...     "web_search": "No",
        ...     "grading_status": "pending",
        ...     "query_rewrite_count": 0,
        ... }
    """
    question: str
    documents: list[Document]
    generation: str
    web_search: str
    grading_status: str
    query_rewrite_count: int


def create_initial_state(question: str) -> AgentState:
    """
    Factory function to create a properly initialized AgentState.
    
    Args:
        question: The user's input query.
    
    Returns:
        A new AgentState with all fields properly initialized.
    """
    return AgentState(
        question=question,
        documents=[],
        generation="",
        web_search="No",
        grading_status="pending",
        query_rewrite_count=0,
    )
