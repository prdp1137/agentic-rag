"""
Qdrant Vector Database Handler.

This module manages the Qdrant connection with Binary Quantization
and Hybrid Search capabilities for high-volume retrieval (10M+ vectors).
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    Distance,
    NamedSparseVector,
    PointStruct,
    VectorParams,
)

from src.config.settings import get_settings
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.sparse_encoder import SparseEncoder


# Global handler instance
_qdrant_handler: QdrantHandler | None = None


@dataclass
class QdrantHandler:
    """
    Manages Qdrant vector database connection with Binary Quantization
    and Hybrid Search capabilities.
    
    This handler is optimized for high-volume retrieval (10M+ vectors) using:
    - Binary Quantization: 32x memory reduction, 40x faster search
    - Hybrid Search: Combines dense (semantic) and sparse (keyword) retrieval
    - Async operations: Non-blocking I/O for production workloads
    
    Attributes:
        collection_name: Name of the Qdrant collection.
        embedding_dim: Dimension of the dense embedding vectors.
        qdrant_url: URL of the Qdrant server.
        qdrant_api_key: API key for Qdrant Cloud (optional for local).
        embedding_model: Name of the embedding model to use.
        client: Async Qdrant client instance.
        embeddings: OpenAI embeddings model for dense vectors.
        sparse_encoder: Encoder for sparse keyword vectors.
    """
    collection_name: str = field(default_factory=lambda: get_settings().qdrant_collection_name)
    embedding_dim: int = field(default_factory=lambda: get_settings().embedding_dimensions)
    qdrant_url: str = field(default_factory=lambda: get_settings().qdrant_url)
    qdrant_api_key: str | None = field(default_factory=lambda: get_settings().qdrant_api_key)
    embedding_model: str = field(default_factory=lambda: get_settings().embedding_model)
    client: AsyncQdrantClient | None = field(default=None, init=False)
    embeddings: OpenAIEmbeddings | None = field(default=None, init=False)
    sparse_encoder: SparseEncoder = field(default_factory=SparseEncoder, init=False)
    
    async def initialize(self) -> None:
        """Initialize async client and embeddings model."""
        settings = get_settings()
        
        self.client = AsyncQdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=settings.qdrant_timeout,
            check_compatibility=False,
        )
        
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            dimensions=self.embedding_dim,
        )
        
        self.sparse_encoder = SparseEncoder()
    
    async def create_collection_with_binary_quantization(self) -> None:
        """
        Create a Qdrant collection with Binary Quantization enabled.
        
        Binary Quantization Configuration:
        - Converts 32-bit floats to 1-bit representations
        - 32x memory reduction
        - always_ram=True: Keeps quantized vectors in RAM for fastest access
        
        This configuration is optimal for:
        - Collections with 10M+ vectors
        - Latency-critical applications
        - Cost-sensitive deployments (reduced RAM requirements)
        """
        if self.client is None:
            raise RuntimeError("Client not initialized. Call initialize() first.")
        
        # Check if collection already exists
        collections = await self.client.get_collections()
        existing_names = [c.name for c in collections.collections]
        
        if self.collection_name in existing_names:
            print(f"Collection '{self.collection_name}' already exists. Skipping creation.")
            return
        
        # Binary Quantization configuration for speed optimization
        # This is the KEY configuration for high-volume vector search
        quantization_config = BinaryQuantization(
            binary=BinaryQuantizationConfig(
                always_ram=True  # Keep quantized vectors in RAM for fastest access
            )
        )
        
        # Create collection with both dense and sparse vector support
        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                # Dense vectors for semantic similarity (cosine)
                "dense": VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                    on_disk=False,  # Keep in memory for speed
                ),
            },
            # Sparse vectors configuration for BM25/SPLADE-like keyword matching
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,  # Use IDF weighting for better keyword matching
                )
            },
            quantization_config=quantization_config,
            # Optimizations for high-volume search
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20000,  # Start indexing after 20k points
                memmap_threshold=50000,    # Memory-map after 50k points
            ),
        )
        print(f"Created collection '{self.collection_name}' with Binary Quantization enabled.")
    
    async def add_documents(self, documents: list[Document]) -> None:
        """
        Add documents to the collection with both dense and sparse vectors.
        
        Args:
            documents: List of LangChain Document objects to index.
        """
        if self.client is None or self.embeddings is None:
            raise RuntimeError("Client not initialized. Call initialize() first.")
        
        # Generate dense embeddings for all documents
        texts = [doc.page_content for doc in documents]
        dense_embeddings = await self.embeddings.aembed_documents(texts)
        
        # Create points with both dense and sparse vectors
        points: list[PointStruct] = []
        for doc, dense_vec in zip(documents, dense_embeddings):
            sparse_vec = self.sparse_encoder.encode(doc.page_content)
            
            # Qdrant expects named vectors as a dict where:
            # - Dense vectors are passed as lists of floats
            # - Sparse vectors are passed as SparseVector objects
            # The SparseVector object will be properly serialized by Qdrant
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vec,  # List of floats
                    "sparse": sparse_vec,  # SparseVector object
                },
                payload={
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "unknown"),
                },
            )
            points.append(point)
        
        # Batch upsert for efficiency
        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        print(f"Indexed {len(points)} documents to '{self.collection_name}'.")
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[Document]:
        """
        Perform Hybrid Search combining Dense and Sparse retrieval with RRF fusion.
        
        This method executes two parallel queries:
        1. Dense Search: Cosine similarity on semantic embeddings
        2. Sparse Search: Keyword matching using sparse vectors (BM25-like)
        
        Results are fused using Reciprocal Rank Fusion (RRF) to combine
        the strengths of both retrieval methods.
        
        Args:
            query: The search query string.
            top_k: Number of results to return after fusion.
            score_threshold: Minimum score threshold for dense search.
        
        Returns:
            List of Document objects ranked by fused relevance score.
        """
        if self.client is None or self.embeddings is None:
            raise RuntimeError("Client not initialized. Call initialize() first.")
        
        settings = get_settings()
        top_k = top_k or settings.retrieval_top_k
        score_threshold = score_threshold or settings.retrieval_score_threshold
        
        # Generate query embeddings
        query_dense = await self.embeddings.aembed_query(query)
        query_sparse = self.sparse_encoder.encode(query)
        
        # Execute Dense and Sparse searches in parallel
        # For named vectors, use tuple format: (vector_name, vector)
        # Dense vectors are passed as lists
        dense_task = self.client.search(
            collection_name=self.collection_name,
            query_vector=("dense", query_dense),
            limit=top_k * 2,  # Fetch more for fusion
            score_threshold=score_threshold,
            with_payload=True,
        )
        
        # For sparse vector queries with named vectors, use NamedSparseVector
        # This is the correct format for querying sparse named vectors
        sparse_task = self.client.search(
            collection_name=self.collection_name,
            query_vector=NamedSparseVector(
                name="sparse",
                vector=query_sparse,
            ),
            limit=top_k * 2,
            with_payload=True,
        )
        
        # Await both searches concurrently
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
        
        # Apply Reciprocal Rank Fusion to merge results
        fused_results = reciprocal_rank_fusion(
            results=[dense_results, sparse_results],
            k=settings.rrf_k,
        )
        
        # Convert to LangChain Documents
        documents: list[Document] = []
        for point in fused_results[:top_k]:
            if point.payload:
                doc = Document(
                    page_content=point.payload.get("content", ""),
                    metadata={
                        **point.payload.get("metadata", {}),
                        "score": point.score,
                        "source": point.payload.get("source", "qdrant"),
                    },
                )
                documents.append(doc)
        
        return documents
    
    async def close(self) -> None:
        """Close the Qdrant client connection."""
        if self.client:
            await self.client.close()


async def get_qdrant_handler() -> QdrantHandler:
    """
    Get or create the global Qdrant handler instance.
    
    Uses a singleton pattern to ensure only one connection is maintained.
    
    Returns:
        Initialized QdrantHandler instance.
    """
    global _qdrant_handler
    if _qdrant_handler is None:
        _qdrant_handler = QdrantHandler()
        await _qdrant_handler.initialize()
    return _qdrant_handler


async def cleanup_qdrant_handler() -> None:
    """Cleanup the global Qdrant handler."""
    global _qdrant_handler
    if _qdrant_handler:
        await _qdrant_handler.close()
        _qdrant_handler = None
