"""
Retrieve Node.

This node performs hybrid search retrieval from Qdrant using
Dense + Sparse vectors with Reciprocal Rank Fusion, followed by
optional advanced re-ranking (cross-encoder or Cohere).
"""

from __future__ import annotations

import os

from src.retrieval.qdrant_handler import get_qdrant_handler
from src.retrieval.reranker import get_reranker
from src.state.agent_state import AgentState


async def retrieve(state: AgentState) -> AgentState:
    """
    Retrieve documents using Hybrid Search with RRF fusion and optional reranking.
    
    Pipeline:
    1. Hybrid Search (Dense + Sparse) with RRF fusion
    2. Optional Cross-encoder or Cohere reranking
    
    The hybrid approach ensures:
    - Semantic understanding via dense embeddings
    - Keyword precision via sparse vectors
    - Robust ranking via RRF fusion
    - Optional: Refined ranking via cross-encoder reranking
    
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
        print(f"   📥 Retrieved {len(documents)} documents via Hybrid+RRF search")
        
        reranker_type = os.getenv("RERANKER_TYPE", "none")
        if reranker_type != "none" and documents:
            reranker = get_reranker(reranker_type)
            print(f"   🔄 Reranking with {reranker_type}...")
            
            documents = await reranker.rerank(
                query=state["question"],
                documents=documents,
                top_k=len(documents),
            )
            print(f"   ✅ Reranked {len(documents)} documents")
        
    except Exception as e:
        print(f"   ⚠️ Retrieval error: {e}")
        documents = []
    
    return {
        **state,
        "documents": documents,
        "grading_status": "pending",
    }
