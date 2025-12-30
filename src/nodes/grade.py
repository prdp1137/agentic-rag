"""
Grade Documents Node.

This node implements the Corrective RAG pattern by evaluating
document relevance and filtering irrelevant results.

Supports multiple grading strategies:
- parallel: Grade all docs in parallel (fast, recommended)
- batch: Grade all docs in a single LLM call (cheapest)
- skip: Skip grading, use retrieval score only (fastest, no LLM cost)
"""

from __future__ import annotations

import asyncio
import os

from langchain_core.documents import Document

from src.chains.grader import get_grader_chain, get_batch_grader_chain
from src.state.agent_state import AgentState


# Grading strategy: "parallel" | "batch" | "skip"
GRADING_STRATEGY = os.getenv("GRADING_STRATEGY", "parallel")

# Minimum score threshold when skipping LLM grading
SCORE_THRESHOLD_FOR_SKIP = float(os.getenv("SCORE_THRESHOLD_FOR_SKIP", "0.01"))


async def grade_single_document(
    grader_chain,
    question: str,
    doc: Document,
    index: int,
) -> tuple[int, Document | None, str]:
    """Grade a single document and return result."""
    try:
        doc_content = doc.page_content[:2000]
        
        grade = await grader_chain.ainvoke({
            "question": question,
            "document": doc_content,
        })
        
        is_relevant = grade.strip().lower() == "yes"
        score = doc.metadata.get("score", "N/A")
        if isinstance(score, float):
            score = f"{score:.3f}"
        
        status = "✅ Relevant" if is_relevant else "❌ Irrelevant"
        
        return (index, doc if is_relevant else None, f"Doc {index+1}: {status} (score: {score})")
    except Exception as e:
        return (index, doc, f"Doc {index+1}: ⚠️ Error, including anyway: {e}")


async def grade_documents_parallel(state: AgentState) -> AgentState:
    """Grade documents in parallel using asyncio.gather."""
    print(f"\n📋 GRADE [parallel]: Evaluating {len(state['documents'])} documents")
    
    grader_chain = get_grader_chain()
    
    # Create tasks for parallel execution
    tasks = [
        grade_single_document(grader_chain, state["question"], doc, i)
        for i, doc in enumerate(state["documents"])
    ]
    
    # Execute all grading in parallel
    results = await asyncio.gather(*tasks)
    
    # Sort by index and collect results
    results.sort(key=lambda x: x[0])
    
    relevant_docs: list[Document] = []
    for index, doc, message in results:
        print(f"   {message}")
        if doc is not None:
            relevant_docs.append(doc)
    
    return _finalize_grading(state, relevant_docs)


async def grade_documents_batch(state: AgentState) -> AgentState:
    """Grade all documents in a single LLM call (most cost-effective)."""
    print(f"\n📋 GRADE [batch]: Evaluating {len(state['documents'])} documents in single call")
    
    batch_grader = get_batch_grader_chain()
    
    # Format all documents for batch grading
    docs_text = ""
    for i, doc in enumerate(state["documents"]):
        score = doc.metadata.get("score", "N/A")
        if isinstance(score, float):
            score = f"{score:.3f}"
        docs_text += f"\n[Document {i+1}] (score: {score})\n{doc.page_content[:1000]}\n"
    
    try:
        result = await batch_grader.ainvoke({
            "question": state["question"],
            "documents": docs_text,
            "num_docs": len(state["documents"]),
        })
        
        # Parse result - expect comma-separated indices like "1,3,4" or "none"
        result = result.strip().lower()
        
        if result == "none" or not result:
            relevant_indices = set()
        else:
            try:
                relevant_indices = {int(x.strip()) - 1 for x in result.split(",") if x.strip().isdigit()}
            except ValueError:
                relevant_indices = set(range(len(state["documents"])))  # Include all on parse error
        
        relevant_docs = []
        for i, doc in enumerate(state["documents"]):
            is_relevant = i in relevant_indices
            score = doc.metadata.get("score", "N/A")
            if isinstance(score, float):
                score = f"{score:.3f}"
            status = "✅ Relevant" if is_relevant else "❌ Irrelevant"
            print(f"   Doc {i+1}: {status} (score: {score})")
            if is_relevant:
                relevant_docs.append(doc)
        
    except Exception as e:
        print(f"   ⚠️ Batch grading error: {e}, including all docs")
        relevant_docs = state["documents"]
    
    return _finalize_grading(state, relevant_docs)


async def grade_documents_skip(state: AgentState) -> AgentState:
    """Skip LLM grading - use retrieval score threshold only (fastest, no cost)."""
    print(f"\n📋 GRADE [skip]: Using score threshold only (no LLM calls)")
    
    relevant_docs = []
    for i, doc in enumerate(state["documents"]):
        score = doc.metadata.get("score", 0)
        if isinstance(score, (int, float)) and score >= SCORE_THRESHOLD_FOR_SKIP:
            relevant_docs.append(doc)
            print(f"   Doc {i+1}: ✅ Score {score:.3f} >= {SCORE_THRESHOLD_FOR_SKIP}")
        else:
            score_str = f"{score:.3f}" if isinstance(score, float) else str(score)
            print(f"   Doc {i+1}: ❌ Score {score_str} < {SCORE_THRESHOLD_FOR_SKIP}")
    
    return _finalize_grading(state, relevant_docs)


def _finalize_grading(state: AgentState, relevant_docs: list[Document]) -> AgentState:
    """Finalize grading result and determine next action."""
    if len(relevant_docs) == 0:
        print("   ⚠️ No relevant documents found. Triggering web search fallback.")
        web_search = "Yes"
        grading_status = "irrelevant"
    else:
        print(f"   ✅ {len(relevant_docs)} relevant documents retained")
        web_search = "No"
        grading_status = "relevant"
    
    return {
        **state,
        "documents": relevant_docs,
        "web_search": web_search,
        "grading_status": grading_status,
    }


async def grade_documents(state: AgentState) -> AgentState:
    """
    Grade retrieved documents for relevance.
    
    Strategy is controlled by GRADING_STRATEGY env var:
    - "parallel": Grade all docs in parallel (default, fast)
    - "batch": Grade all docs in single LLM call (cheapest)  
    - "skip": Skip LLM grading, use score threshold (fastest, free)
    
    Args:
        state: Current agent state with documents to grade.
    
    Returns:
        Updated state with filtered documents and web_search flag.
    """
    if not state["documents"]:
        print("\n📋 GRADE: No documents to grade")
        return {
            **state,
            "web_search": "Yes",
            "grading_status": "irrelevant",
        }
    
    strategy = GRADING_STRATEGY.lower()
    
    if strategy == "skip":
        return await grade_documents_skip(state)
    elif strategy == "batch":
        return await grade_documents_batch(state)
    else:  # default: parallel
        return await grade_documents_parallel(state)
