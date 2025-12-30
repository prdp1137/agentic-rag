"""
Answer Generation Chain.

This module provides the final answer generation chain that
synthesizes responses from retrieved context documents.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from src.chains.llm import get_llm


GENERATOR_SYSTEM_PROMPT = """You are a knowledgeable AI assistant.
Answer the user's question based on the provided context documents.

Guidelines:
- Use information from the context to formulate your answer
- If the context doesn't contain enough information, acknowledge the limitation
- Be concise but comprehensive
- Cite relevant parts of the context when appropriate
- Structure your answer for readability
- If multiple sources provide different perspectives, synthesize them
- Do not make up information not present in the context
- Always use the context provided below to answer the question"""


GENERATOR_HUMAN_TEMPLATE = """Below are context documents that contain information relevant to the question.

{context}

Based on the context documents above, answer the following question:

{question}

Provide a comprehensive answer using the information from the context documents:"""


def get_generation_chain() -> Runnable:
    """
    Create the final answer generation chain.
    
    This chain synthesizes a comprehensive answer using the
    filtered relevant documents as context.
    
    The generator is designed to:
    - Ground answers in the provided context
    - Acknowledge uncertainty when appropriate
    - Provide well-structured responses
    
    Returns:
        A LangChain Runnable that outputs the final answer.
    
    Example:
        >>> chain = get_generation_chain()
        >>> result = await chain.ainvoke({
        ...     "context": "[Document 1] LangGraph enables...",
        ...     "question": "What is LangGraph?"
        ... })
        >>> print(result)  # Comprehensive answer based on context
    """
    # Use tuple format to enable template variable substitution
    generation_prompt = ChatPromptTemplate.from_messages([
        ("system", GENERATOR_SYSTEM_PROMPT),
        ("human", GENERATOR_HUMAN_TEMPLATE),
    ])
    
    llm = get_llm()
    
    return generation_prompt | llm | StrOutputParser()


def format_documents_as_context(documents: list) -> str:
    """
    Format a list of documents into a context string for generation.
    
    Args:
        documents: List of Document objects.
    
    Returns:
        Formatted string with numbered documents and sources.
    """
    if not documents:
        return "No context documents are available."
    
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        score = doc.metadata.get("score", "N/A")
        if isinstance(score, float):
            score = f"{score:.3f}"
        
        # Ensure we have actual content
        content = doc.page_content.strip() if doc.page_content else "[Empty document]"
        
        context_parts.append(
            f"--- Document {i} (Source: {source}, Relevance Score: {score}) ---\n{content}"
        )
    
    formatted_context = "\n\n".join(context_parts)
    
    # Ensure we return a non-empty string
    if not formatted_context or formatted_context.strip() == "":
        return "No context documents are available."
    
    return formatted_context
