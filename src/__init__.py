"""
Multi-Agent RAG System with Corrective RAG (CRAG) Capabilities.

This package provides a production-ready implementation of an Autonomous
Enterprise RAG system using LangGraph for orchestration and Qdrant for
high-volume vector retrieval.
"""

from src.graph.builder import build_crag_graph, compile_graph_with_persistence
from src.state.agent_state import AgentState

__version__ = "1.0.0"
__all__ = [
    "AgentState",
    "build_crag_graph",
    "compile_graph_with_persistence",
]
