"""
Graph Builder for the Corrective RAG System.

This module constructs and compiles the LangGraph StateGraph
with all nodes, edges, and persistence configuration.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.config.settings import get_settings
from src.graph.routing import route_after_grading
from src.nodes import generate, grade_documents, retrieve, transform_query, web_search
from src.state.agent_state import AgentState


def build_crag_graph() -> StateGraph:
    """
    Build the Corrective RAG graph with all nodes and edges.
    
    Graph Architecture:
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     CORRECTIVE RAG GRAPH                            │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │                         ┌──────────────┐                            │
    │                         │   RETRIEVE   │ ◄── Entry Point            │
    │                         │  (Hybrid+RRF)│                            │
    │                         └──────┬───────┘                            │
    │                                │                                    │
    │                                ▼                                    │
    │                      ┌──────────────────┐                           │
    │                      │ GRADE_DOCUMENTS  │                           │
    │                      │   (LLM Grader)   │                           │
    │                      └────────┬─────────┘                           │
    │                               │                                     │
    │                    ┌──────────┴──────────┐                          │
    │                    │  CONDITIONAL EDGE   │                          │
    │                    │  (Correction Loop)  │                          │
    │                    └──────────┬──────────┘                          │
    │                               │                                     │
    │              ┌────────────────┼────────────────┐                    │
    │              │                │                │                    │
    │              ▼                ▼                │                    │
    │    ┌─────────────────┐  ┌──────────┐          │                    │
    │    │ TRANSFORM_QUERY │  │ GENERATE │          │                    │
    │    │   (Rewrite)     │  │ (GPT-4o) │          │                    │
    │    └────────┬────────┘  └────┬─────┘          │                    │
    │             │                │                │                    │
    │             ▼                ▼                │                    │
    │    ┌─────────────────┐  ┌──────────┐          │                    │
    │    │   WEB_SEARCH    │  │   END    │          │                    │
    │    │   (Fallback)    │  └──────────┘          │                    │
    │    └────────┬────────┘                        │                    │
    │             │                                 │                    │
    │             ▼                                 │                    │
    │    ┌─────────────────┐                        │                    │
    │    │    GENERATE     │ ◄──────────────────────┘                    │
    │    └────────┬────────┘                                             │
    │             │                                                      │
    │             ▼                                                      │
    │    ┌─────────────────┐                                             │
    │    │      END        │                                             │
    │    └─────────────────┘                                             │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    
    Returns:
        Configured StateGraph ready for compilation.
    """
    # Initialize the StateGraph with our state schema
    workflow = StateGraph(AgentState)
    
    # -------------------------------------------------------------------------
    # Add all nodes to the graph
    # -------------------------------------------------------------------------
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    
    # -------------------------------------------------------------------------
    # Set the entry point
    # -------------------------------------------------------------------------
    workflow.set_entry_point("retrieve")
    
    # -------------------------------------------------------------------------
    # Add edges for the flow
    # -------------------------------------------------------------------------
    
    # Edge 1: retrieve → grade_documents (always)
    # After retrieval, we always grade the documents for relevance
    workflow.add_edge("retrieve", "grade_documents")
    
    # Edge 2: grade_documents → conditional routing
    # This is the "Correction Loop" implementation
    # The routing function decides whether to:
    # - Proceed to generation (if documents are relevant)
    # - Enter correction loop (if documents are irrelevant)
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "transform_query": "transform_query",  # Correction path
            "generate": "generate",                 # Happy path
        }
    )
    
    # Edge 3: transform_query → web_search
    # After query rewriting, we attempt web search as fallback
    workflow.add_edge("transform_query", "web_search")
    
    # Edge 4: web_search → generate
    # After web search, we always proceed to generation
    workflow.add_edge("web_search", "generate")
    
    # Edge 5: generate → END
    # Generation is the terminal node
    workflow.add_edge("generate", END)
    
    return workflow


def compile_graph_with_persistence(
    use_postgres: bool | None = None,
    postgres_conn_string: str | None = None,
) -> Any:
    """
    Compile the graph with persistence for Time Travel and state recovery.
    
    Persistence Benefits:
    - State checkpointing at each node
    - "Time Travel" debugging (replay from any checkpoint)
    - Crash recovery (resume from last checkpoint)
    - Multi-turn conversations with memory
    - Audit trail for compliance
    
    Args:
        use_postgres: If True, use PostgresSaver. If None, uses settings.
        postgres_conn_string: PostgreSQL connection string. If None, uses settings.
    
    Returns:
        Compiled graph with persistence enabled.
    
    Example:
        >>> # Development mode with memory persistence
        >>> app = compile_graph_with_persistence()
        
        >>> # Production mode with PostgreSQL
        >>> app = compile_graph_with_persistence(
        ...     use_postgres=True,
        ...     postgres_conn_string="postgresql://user:pass@localhost/db"
        ... )
    """
    settings = get_settings()
    
    # Determine persistence configuration
    if use_postgres is None:
        use_postgres = settings.persistence_type == "postgres"
    
    if postgres_conn_string is None:
        postgres_conn_string = settings.postgres_connection_string
    
    # Build the workflow
    workflow = build_crag_graph()
    
    # Configure checkpointer
    if use_postgres and postgres_conn_string:
        # PostgreSQL persistence for production
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            
            checkpointer = AsyncPostgresSaver.from_conn_string(postgres_conn_string)
            print("✅ Using PostgresSaver for persistence (production mode)")
        except ImportError:
            print("⚠️ PostgresSaver not available. Install: pip install langgraph-checkpoint-postgres")
            print("   Falling back to MemorySaver.")
            checkpointer = MemorySaver()
    else:
        # In-memory persistence for demo/development
        checkpointer = MemorySaver()
        print("✅ Using MemorySaver for persistence (demo mode)")
    
    # Compile with checkpointer
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    
    return compiled_graph
