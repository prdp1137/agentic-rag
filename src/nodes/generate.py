"""
Generate Node.

This node synthesizes the final answer using the
filtered relevant documents as context.
"""

from __future__ import annotations

from src.chains.generator import format_documents_as_context, get_generation_chain
from src.state.agent_state import AgentState


async def generate(state: AgentState) -> AgentState:
    """
    Generate the final answer using filtered relevant documents.
    
    This node synthesizes a comprehensive answer from the context
    documents (from vector DB and/or web search).
    
    The generation process:
    1. Format documents into a structured context
    2. Pass context and question to the LLM
    3. Generate a grounded, helpful response
    
    Args:
        state: Current agent state with filtered documents.
    
    Returns:
        Updated state with the generated answer.
    """
    print(f"\n💡 GENERATE: Synthesizing answer from {len(state['documents'])} documents")
    
    generation_chain = get_generation_chain()
    
    # Validate documents have content
    valid_documents = [
        doc for doc in state["documents"]
        if doc and hasattr(doc, 'page_content') and doc.page_content and doc.page_content.strip()
    ]
    
    if len(valid_documents) != len(state["documents"]):
        print(f"   ⚠️ Warning: {len(state['documents']) - len(valid_documents)} documents have no content")
    
    # Format context from documents
    context = format_documents_as_context(valid_documents)
    question = state["question"]
    
    # Debug: Print context length and question
    print(f"   📄 Context length: {len(context)} characters")
    print(f"   📚 Valid documents: {len(valid_documents)}/{len(state['documents'])}")
    print(f"   ❓ Question: {question[:100]}..." if len(question) > 100 else f"   ❓ Question: {question}")
    
    # Ensure we have valid inputs
    if not context or context == "No context documents are available.":
        print("   ⚠️ Warning: No context available for generation")
        return {
            **state,
            "generation": (
                "I apologize, but I could not find any relevant context documents to answer your question. "
                "Please try rephrasing your question or ensure that relevant documents are available in the knowledge base."
            ),
        }
    
    if not question or not question.strip():
        print("   ⚠️ Warning: No question provided")
        return {
            **state,
            "generation": "Error: No question provided for generation.",
        }
    
    try:
        generation = await generation_chain.ainvoke({
            "context": context,
            "question": question,
        })
        print(f"   ✅ Answer generated successfully ({len(generation)} characters)")
    except Exception as e:
        print(f"   ⚠️ Generation error: {e}")
        import traceback
        traceback.print_exc()
        generation = (
            f"I apologize, but I encountered an error generating the answer: {str(e)}\n\n"
            f"Please try again or rephrase your question."
        )
    
    return {
        **state,
        "generation": generation,
    }
