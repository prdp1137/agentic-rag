"""
Advanced Re-ranking Module.

Provides cross-encoder and LLM-based reranking for improved relevance.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseReranker(ABC):
    """Abstract base class for rerankers."""
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        """Rerank documents by relevance to query."""
        pass


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker using sentence-transformers.
    
    Cross-encoders score query-document pairs together, providing
    more accurate relevance scores than bi-encoders (embeddings).
    
    Models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (slower, better quality)
    - BAAI/bge-reranker-base (good multilingual support)
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
    
    def _get_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for cross-encoder reranking. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    async def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        """
        Rerank documents using cross-encoder model.
        
        Args:
            query: The search query.
            documents: List of documents to rerank.
            top_k: Number of top documents to return.
        
        Returns:
            Reranked list of documents with updated scores.
        """
        if not documents:
            return []
        
        model = self._get_model()
        
        pairs = [(query, doc.page_content[:512]) for doc in documents]
        
        scores = model.predict(pairs)
        
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        reranked = []
        for doc, score in scored_docs[:top_k]:
            doc.metadata["rerank_score"] = float(score)
            doc.metadata["original_score"] = doc.metadata.get("score", 0)
            doc.metadata["score"] = float(score)  # Update main score
            reranked.append(doc)
        
        return reranked


class CohereReranker(BaseReranker):
    """
    Cohere Rerank API-based reranker.
    
    Uses Cohere's neural reranking model for high-quality results.
    Requires COHERE_API_KEY environment variable.
    """
    
    def __init__(self, model: str = "rerank-english-v3.0"):
        self.model = model
        self._client = None
    
    def _get_client(self):
        """Lazy load the Cohere client."""
        if self._client is None:
            try:
                import cohere
                api_key = os.getenv("COHERE_API_KEY")
                if not api_key:
                    raise ValueError("COHERE_API_KEY environment variable required")
                self._client = cohere.Client(api_key)
            except ImportError:
                raise ImportError(
                    "cohere package required for Cohere reranking. "
                    "Install with: pip install cohere"
                )
        return self._client
    
    async def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        """
        Rerank documents using Cohere Rerank API.
        
        Args:
            query: The search query.
            documents: List of documents to rerank.
            top_k: Number of top documents to return.
        
        Returns:
            Reranked list of documents with updated scores.
        """
        if not documents:
            return []
        
        client = self._get_client()
        
        # Extract document texts
        doc_texts = [doc.page_content[:4096] for doc in documents]
        
        # Call Cohere rerank
        response = client.rerank(
            query=query,
            documents=doc_texts,
            top_n=top_k,
            model=self.model,
        )
        
        # Build reranked list
        reranked = []
        for result in response.results:
            doc = documents[result.index]
            doc.metadata["rerank_score"] = result.relevance_score
            doc.metadata["original_score"] = doc.metadata.get("score", 0)
            doc.metadata["score"] = result.relevance_score
            reranked.append(doc)
        
        return reranked


class NoOpReranker(BaseReranker):
    """No-op reranker that returns documents unchanged."""
    
    async def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        """Return documents as-is, just limiting to top_k."""
        return documents[:top_k]


def get_reranker(reranker_type: str | None = None) -> BaseReranker:
    """
    Factory function to get a reranker.
    
    Args:
        reranker_type: Type of reranker. Options:
            - "cross-encoder": Local cross-encoder model (free, needs GPU for speed)
            - "cohere": Cohere Rerank API (paid, fast)
            - "none" or None: No reranking
    
    Returns:
        Configured reranker instance.
    """
    reranker_type = reranker_type or os.getenv("RERANKER_TYPE", "none")
    
    if reranker_type == "cross-encoder":
        model = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        return CrossEncoderReranker(model_name=model)
    elif reranker_type == "cohere":
        model = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")
        return CohereReranker(model=model)
    else:
        return NoOpReranker()
