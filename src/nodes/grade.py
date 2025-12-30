"""
Grade Documents Node.

This node implements the Corrective RAG pattern by evaluating
document relevance and filtering irrelevant results.
"""

from __future__ import annotations

from langchain_core.documents import Document

from src.chains.grader import get_grader_chain
from src.state.agent_state import AgentState


async def grade_documents(state: AgentState) -> AgentState:
    """
    Grade retrieved documents for relevance using LLM.
    
    This node implements the Corrective RAG pattern by evaluating
    each document's relevance to the question. Irrelevant documents
    are filtered out. If ALL documents are irrelevant, web search
    fallback is triggered.
    
    The grading process:
    1. Iterate through each retrieved document
    2. Use LLM to evaluate relevance (yes/no)
    3. Filter out irrelevant documents
    4. Set web_search flag if all docs are irrelevant
    
    Args:
        state: Current agent state with documents to grade.
    
    Returns:
        Updated state with filtered documents and web_search flag.
    """
    print(f"\n📋 GRADE: Evaluating {len(state['documents'])} documents for relevance")
    
    grader_chain = get_grader_chain()
    relevant_docs: list[Document] = []
    
    for i, doc in enumerate(state["documents"]):
        try:
            # Truncate document content for LLM context efficiency
            doc_content = doc.page_content[:2000]
            
            grade = await grader_chain.ainvoke({
                "question": state["question"],
                "document": doc_content,
            })
            
            is_relevant = grade.strip().lower() == "yes"
            score = doc.metadata.get("score", "N/A")
            if isinstance(score, float):
                score = f"{score:.3f}"
            
            status = "✅ Relevant" if is_relevant else "❌ Irrelevant"
            print(f"   Doc {i+1}: {status} (score: {score})")
            
            if is_relevant:
                relevant_docs.append(doc)
                
        except Exception as e:
            print(f"   ⚠️ Grading error for doc {i+1}: {e}")
            # On error, include the document (fail-safe approach)
            relevant_docs.append(doc)
    
    # Determine if web search is needed
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
