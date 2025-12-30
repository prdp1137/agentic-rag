"""
Retrieve Node.

This node performs hybrid search retrieval from Qdrant using
Dense + Sparse vectors with Reciprocal Rank Fusion.
"""

from __future__ import annotations

from src.retrieval.qdrant_handler import get_qdrant_handler
from src.state.agent_state import AgentState


async def retrieve(state: AgentState) -> AgentState:
    """
    Retrieve documents using Hybrid Search with RRF fusion.
    
    This node fetches relevant documents from Qdrant using both
    dense (semantic) and sparse (keyword) retrieval methods,
    then fuses the results using Reciprocal Rank Fusion.
    
    The hybrid approach ensures:
    - Semantic understanding via dense embeddings
    - Keyword precision via sparse vectors
    - Robust ranking via RRF fusion
    
    Args:
        state: Current agent state containing the question.
    
    Returns:
        Updated state with retrieved documents.
    """
    print(f"\n🔍 RETRIEVE: Fetching documents for query: '{state['question']}'")
    
    handler = await get_qdrant_handler()
    
    try:
        documents = await handler.hybrid_search(
            query=state["question"],
        )
        print(f"   Retrieved {len(documents)} documents via Hybrid+RRF search")
    except Exception as e:
        print(f"   ⚠️ Retrieval error: {e}")
        documents = []
    
    return {
        **state,
        "documents": documents,
        "grading_status": "pending",
    }
