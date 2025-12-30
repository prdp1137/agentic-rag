"""
Reciprocal Rank Fusion (RRF) Algorithm.

This module implements the RRF algorithm for merging ranked results
from multiple retrieval methods (e.g., dense vectors + sparse BM25).
"""

from __future__ import annotations

from collections import defaultdict

from qdrant_client.models import ScoredPoint


def reciprocal_rank_fusion(
    results: list[list[ScoredPoint]],
    k: int = 60
) -> list[ScoredPoint]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.
    
    RRF is a robust fusion algorithm that combines rankings from different
    retrieval methods (e.g., dense vectors + sparse BM25) without requiring
    score normalization. It's particularly effective for hybrid search.
    
    Formula: RRF(d) = Σ 1/(k + rank(d))
    
    Where k is a constant (typically 60) that mitigates the impact of
    high rankings from individual retrieval methods.
    
    Args:
        results: List of ranked result lists, where each inner list contains
                 ScoredPoint objects from a different retrieval method.
                 Example: [dense_results, sparse_results]
        k: Fusion constant that controls ranking sensitivity. Higher k
           reduces the impact of rank differences. Default is 60 (standard).
    
    Returns:
        A single merged list of ScoredPoint objects, sorted by fused score
        in descending order.
    
    Example:
        >>> dense_results = [point1, point2, point3]  # From cosine similarity
        >>> sparse_results = [point2, point1, point4]  # From BM25
        >>> fused = reciprocal_rank_fusion([dense_results, sparse_results])
        >>> # point2 ranks higher due to appearing at top in both lists
    
    References:
        - Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
          "Reciprocal rank fusion outperforms condorcet and individual
          rank learning methods."
    """
    # Dictionary to accumulate RRF scores for each document
    fused_scores: dict[str, float] = defaultdict(float)
    
    # Map to store point objects by their ID for final result construction
    point_map: dict[str, ScoredPoint] = {}
    
    for result_list in results:
        for rank, point in enumerate(result_list, start=1):
            # Extract document ID - handle both string and UUID types
            doc_id = str(point.id)
            
            # Apply RRF formula: 1 / (k + rank)
            # This gives higher weight to top-ranked documents
            # while preventing any single ranking from dominating
            fused_scores[doc_id] += 1.0 / (k + rank)
            
            # Store the point object (latest version wins, but they should be identical)
            point_map[doc_id] = point
    
    # Sort documents by fused score in descending order
    sorted_doc_ids = sorted(
        fused_scores.keys(),
        key=lambda doc_id: fused_scores[doc_id],
        reverse=True
    )
    
    # Reconstruct the result list with updated scores
    fused_results: list[ScoredPoint] = []
    for doc_id in sorted_doc_ids:
        point = point_map[doc_id]
        # Create a new ScoredPoint with the fused score
        fused_point = ScoredPoint(
            id=point.id,
            version=point.version,
            score=fused_scores[doc_id],
            payload=point.payload,
            vector=point.vector,
        )
        fused_results.append(fused_point)
    
    return fused_results
