"""
Web Search Node.

This node provides fallback web search when vector DB
retrieval fails to find relevant documents.
"""

from __future__ import annotations

import os

from langchain_core.documents import Document

from src.state.agent_state import AgentState


async def web_search(state: AgentState) -> AgentState:
    """
    Fallback web search when vector DB retrieval fails.
    
    This node is triggered when:
    1. All retrieved documents are graded as irrelevant
    2. Query transformation has been attempted
    
    Current implementation is a STUB. For production, integrate with:
    - Tavily API (AI-optimized search)
    - Google Custom Search API
    - Bing Search API
    - DuckDuckGo API
    
    Args:
        state: Current agent state.
    
    Returns:
        Updated state with web search results as documents.
    """
    print(f"\n🌐 WEB_SEARCH: Performing fallback web search")
    print(f"   Query: '{state['question']}'")
    
    # Check for Tavily API key
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    if tavily_api_key:
        # Production: Use Tavily for real web search
        try:
            web_results = await _tavily_search(state["question"], tavily_api_key)
            print(f"   📥 Retrieved {len(web_results)} results from Tavily")
        except Exception as e:
            print(f"   ⚠️ Tavily search error: {e}")
            web_results = _get_stub_results(state["question"])
    else:
        # Development: Use stub results
        print("   ℹ️ TAVILY_API_KEY not set. Using stub results.")
        web_results = _get_stub_results(state["question"])
    
    # Merge with any existing documents
    merged_docs = state["documents"] + web_results
    
    return {
        **state,
        "documents": merged_docs,
        "web_search": "Completed",
    }


async def _tavily_search(query: str, api_key: str) -> list[Document]:
    """
    Perform actual web search using Tavily API.
    
    Args:
        query: Search query.
        api_key: Tavily API key.
    
    Returns:
        List of Document objects from web search.
    """
    try:
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=3,
            include_answer=True,
        )
        
        documents = []
        
        # Add the AI-generated answer if available
        if response.get("answer"):
            documents.append(Document(
                page_content=response["answer"],
                metadata={
                    "source": "tavily_answer",
                    "type": "ai_generated",
                    "score": 1.0,
                }
            ))
        
        # Add individual search results
        for result in response.get("results", []):
            documents.append(Document(
                page_content=result.get("content", ""),
                metadata={
                    "source": "web_search",
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "score": result.get("score", 0.5),
                }
            ))
        
        return documents
        
    except ImportError:
        raise ImportError(
            "Tavily package not installed. Run: pip install tavily-python"
        )


def _get_stub_results(query: str) -> list[Document]:
    """
    Generate stub web search results for development/testing.
    
    Args:
        query: Search query.
    
    Returns:
        List of simulated Document objects.
    """
    return [
        Document(
            page_content=(
                f"[Web Search Result - Stub]\n"
                f"This is a simulated web search result for: '{query}'\n\n"
                f"In production, this would contain actual web content from "
                f"Tavily, Google, or Bing Search API. Configure TAVILY_API_KEY "
                f"in your environment to enable real web search."
            ),
            metadata={
                "source": "web_search_stub",
                "url": "https://example.com/simulated",
                "score": 0.75,
            }
        ),
    ]
