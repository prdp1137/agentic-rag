"""
Conditional Routing Logic for the CRAG Graph.

This module implements the routing decisions that enable
the Corrective RAG pattern with its correction loop.
"""

from __future__ import annotations

from typing import Literal

from src.config.settings import get_settings
from src.state.agent_state import AgentState


def route_after_grading(state: AgentState) -> Literal["transform_query", "generate"]:
    """
    Conditional routing after document grading.
    
    This function implements the "Correction Loop" logic:
    
    Decision Flow:
    ┌─────────────────────────────────────────────────────────────┐
    │                   ROUTE AFTER GRADING                       │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │   web_search == 'Yes'?                                      │
    │        │                                                    │
    │   ┌────┴────┐                                              │
    │   │  YES    │  NO                                          │
    │   ▼         ▼                                              │
    │   Check     ───────────────────────► generate              │
    │   rewrite                             (happy path)         │
    │   count                                                    │
    │   │                                                        │
    │   ├─ < max ──────────────────────► transform_query         │
    │   │                                 (correction loop)      │
    │   │                                                        │
    │   └─ >= max ─────────────────────► generate                │
    │                                     (give up, proceed)     │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    
    This is the core of the Corrective RAG pattern, enabling the system
    to self-correct when initial retrieval fails.
    
    Args:
        state: Current agent state with grading results.
    
    Returns:
        Name of the next node to execute:
        - "transform_query": Enter correction loop
        - "generate": Proceed to answer generation
    """
    settings = get_settings()
    
    if state.get("web_search") == "Yes":
        # Check rewrite count to prevent infinite loops
        rewrite_count = state.get("query_rewrite_count", 0)
        
        if rewrite_count >= settings.max_query_rewrites:
            print(f"   ⚠️ Max query rewrites ({settings.max_query_rewrites}) reached. Proceeding to generation.")
            return "generate"
        
        # Retrieval failed - enter correction loop
        print("   → Routing to: transform_query (entering correction loop)")
        return "transform_query"
    else:
        # Valid documents exist - proceed to generation
        print("   → Routing to: generate (documents are relevant)")
        return "generate"


def route_after_web_search(state: AgentState) -> Literal["generate"]:
    """
    Routing after web search - always proceeds to generation.
    
    After web search fallback, we always move to the generation phase
    with whatever results were obtained. This is a terminal routing
    decision that ends the correction loop.
    
    Args:
        state: Current agent state after web search.
    
    Returns:
        Always returns "generate".
    """
    print("   → Routing to: generate (post web search)")
    return "generate"
