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
    Create a document relevance grader chain for single document.
    
    Returns:
        A LangChain Runnable that outputs 'yes' or 'no' for relevance.
    """
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", GRADER_SYSTEM_PROMPT),
        ("human", GRADER_HUMAN_TEMPLATE),
    ])
    
    llm = get_llm()
    
    return grader_prompt | llm | StrOutputParser()

BATCH_GRADER_SYSTEM_PROMPT = """You are a document relevance grader.
Your task is to identify which documents are relevant to answering the user's question.

Evaluation criteria:
- A document is relevant if it contains information that helps answer the question
- Partial relevance counts as relevant
- If a document discusses related concepts, mark it as relevant

You will be given multiple documents. Return ONLY the numbers of relevant documents as a comma-separated list.
If no documents are relevant, respond with 'none'.

Examples:
- If documents 1, 3, and 4 are relevant: "1,3,4"
- If only document 2 is relevant: "2"
- If no documents are relevant: "none"

IMPORTANT: Only output the numbers or 'none'. No explanations."""


BATCH_GRADER_HUMAN_TEMPLATE = """Question: {question}

Documents to evaluate:
{documents}

Which document numbers (1 to {num_docs}) are relevant to answering the question?
Return only the numbers as comma-separated list, or 'none' if no documents are relevant:"""


def get_batch_grader_chain() -> Runnable:
    """
    Create a batch document relevance grader chain.
    
    This chain evaluates ALL documents in a single LLM call,
    which is more cost-effective than grading one by one.
    
    Returns:
        A LangChain Runnable that outputs comma-separated relevant doc indices.
    
    Example:
        >>> chain = get_batch_grader_chain()
        >>> result = await chain.ainvoke({
        ...     "question": "What is LangGraph?",
        ...     "documents": "[Doc 1] LangGraph is...\n[Doc 2] Unrelated...",
        ...     "num_docs": 2,
        ... })
        >>> print(result)  # "1"
    """
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", BATCH_GRADER_SYSTEM_PROMPT),
        ("human", BATCH_GRADER_HUMAN_TEMPLATE),
    ])
    
    llm = get_llm()
    
    return grader_prompt | llm | StrOutputParser()
