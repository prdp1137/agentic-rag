"""
Query Rewriter Chain.

This module provides an LLM-based chain for rewriting queries
when initial retrieval fails to find relevant documents.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from src.chains.llm import get_llm


REWRITER_SYSTEM_PROMPT = """You are a query optimization expert.
Your task is to rewrite user questions to be more effective for semantic vector search.

Guidelines:
- Expand abbreviations and acronyms
- Add relevant synonyms and related terms
- Remove ambiguous pronouns (he, she, it, they)
- Make the query more specific and descriptive
- Maintain the original intent and meaning
- Use natural language, not keywords
- Focus on the core concepts being asked about

CRITICAL: Output ONLY the rewritten question text. Do NOT include:
- The phrase "Original question:"
- The phrase "Rewrite this question..."
- Any explanations
- Any quotes around the text
- Any other text besides the rewritten question itself"""


REWRITER_HUMAN_TEMPLATE = """Rewrite this question for better vector search retrieval:

{question}

Rewritten question:"""


def get_query_rewriter_chain() -> Runnable:
    """
    Create a query rewriting chain for failed retrievals.
    
    When all retrieved documents are irrelevant, this chain rewrites
    the query to be more optimized for vector search retrieval.
    
    The rewriter focuses on:
    - Expanding context and specificity
    - Adding semantic richness
    - Removing ambiguity
    
    Returns:
        A LangChain Runnable that outputs an optimized query string.
    
    Example:
        >>> chain = get_query_rewriter_chain()
        >>> result = await chain.ainvoke({
        ...     "question": "How does it work?"
        ... })
        >>> print(result)  # More specific, context-rich query
    """
    rewriter_prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITER_SYSTEM_PROMPT),
        ("human", REWRITER_HUMAN_TEMPLATE),
    ])
    
    llm = get_llm()
    
    return rewriter_prompt | llm | StrOutputParser()
