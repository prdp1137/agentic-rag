"""
Document Relevance Grader Chain.

This module provides an LLM-based chain for evaluating document relevance
to user queries, enabling the Corrective RAG pattern.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from src.chains.llm import get_llm


GRADER_SYSTEM_PROMPT = """You are a document relevance grader.
Your task is to assess whether a retrieved document is relevant to a user question.

Evaluation criteria:
- The document must contain information that helps answer the question
- Partial relevance counts as relevant
- If the document discusses related concepts, mark as relevant
- Focus on semantic relevance, not just keyword matching

Respond with ONLY 'yes' or 'no'. No explanations, no punctuation, just the word."""


GRADER_HUMAN_TEMPLATE = """Question: {question}

Document content:
{document}

Is this document relevant to answering the question?"""


def get_grader_chain() -> Runnable:
    """
    Create a document relevance grader chain.
    
    This chain evaluates whether a retrieved document is relevant to
    the user's question, enabling the Corrective RAG pattern.
    
    The grader uses a binary yes/no output for clear decision-making
    in the correction loop.
    
    Returns:
        A LangChain Runnable that outputs 'yes' or 'no' for relevance.
    
    Example:
        >>> chain = get_grader_chain()
        >>> result = await chain.ainvoke({
        ...     "question": "What is LangGraph?",
        ...     "document": "LangGraph is a library for building stateful agents..."
        ... })
        >>> print(result)  # 'yes'
    """
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", GRADER_SYSTEM_PROMPT),
        ("human", GRADER_HUMAN_TEMPLATE),
    ])
    
    llm = get_llm()
    
    return grader_prompt | llm | StrOutputParser()
