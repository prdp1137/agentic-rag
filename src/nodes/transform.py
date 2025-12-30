"""
Transform Query Node.

This node rewrites queries for better retrieval when
the initial search fails to find relevant documents.
"""

from __future__ import annotations

from src.chains.rewriter import get_query_rewriter_chain
from src.state.agent_state import AgentState


async def transform_query(state: AgentState) -> AgentState:
    """
    Rewrite the query for better retrieval when initial search failed.
    
    This node is triggered when all retrieved documents are irrelevant.
    It uses an LLM to reformulate the query for more effective
    vector search retrieval.
    
    Query transformation strategies:
    - Expand abbreviations and acronyms
    - Add semantic context
    - Remove ambiguous references
    - Increase specificity
    
    Args:
        state: Current agent state with the original question.
    
    Returns:
        Updated state with optimized question and incremented rewrite count.
    """
    print(f"\n🔄 TRANSFORM: Rewriting query for better retrieval")
    print(f"   Original: '{state['question']}'")
    
    rewriter_chain = get_query_rewriter_chain()
    
    try:
        new_question = await rewriter_chain.ainvoke({
            "question": state["question"],
        })
        new_question = new_question.strip()
        print(f"   Rewritten: '{new_question}'")
    except Exception as e:
        print(f"   ⚠️ Query rewrite error: {e}")
        new_question = state["question"]  # Keep original on error
    
    return {
        **state,
        "question": new_question,
        "query_rewrite_count": state.get("query_rewrite_count", 0) + 1,
    }
