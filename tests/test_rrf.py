"""
Tests for the Reciprocal Rank Fusion algorithm.
"""

import pytest
from qdrant_client.models import ScoredPoint

from src.retrieval.rrf import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    """Test suite for RRF algorithm."""

    def _create_scored_point(
        self,
        point_id: str,
        score: float,
        content: str = "test content",
    ) -> ScoredPoint:
        """Helper to create a ScoredPoint for testing."""
        return ScoredPoint(
            id=point_id,
            version=1,
            score=score,
            payload={"content": content},
            vector=None,
        )

    def test_empty_results(self):
        """RRF with empty results should return empty list."""
        result = reciprocal_rank_fusion([])
        assert result == []

    def test_single_result_list(self):
        """RRF with single result list should preserve order."""
        points = [
            self._create_scored_point("a", 0.9),
            self._create_scored_point("b", 0.8),
            self._create_scored_point("c", 0.7),
        ]
        
        result = reciprocal_rank_fusion([points])
        
        # Order should be preserved
        assert len(result) == 3
        assert str(result[0].id) == "a"
        assert str(result[1].id) == "b"
        assert str(result[2].id) == "c"

    def test_two_identical_rankings(self):
        """RRF with identical rankings should preserve order."""
        points = [
            self._create_scored_point("a", 0.9),
            self._create_scored_point("b", 0.8),
        ]
        
        result = reciprocal_rank_fusion([points, points])
        
        assert len(result) == 2
        assert str(result[0].id) == "a"
        assert str(result[1].id) == "b"

    def test_different_rankings_fusion(self):
        """RRF should boost documents appearing in multiple lists."""
        # List 1: a > b > c
        list1 = [
            self._create_scored_point("a", 0.9),
            self._create_scored_point("b", 0.8),
            self._create_scored_point("c", 0.7),
        ]
        
        # List 2: b > a > d
        list2 = [
            self._create_scored_point("b", 0.95),
            self._create_scored_point("a", 0.85),
            self._create_scored_point("d", 0.75),
        ]
        
        result = reciprocal_rank_fusion([list1, list2], k=60)
        
        # Both a and b appear in both lists
        # b: rank 2 in list1, rank 1 in list2 -> 1/62 + 1/61
        # a: rank 1 in list1, rank 2 in list2 -> 1/61 + 1/62
        # They should have same score, but b appears first in list2
        assert len(result) == 4
        
        # b should rank higher (rank 1 + rank 2) vs a (rank 1 + rank 2)
        # Actually they have same RRF score, order depends on iteration
        result_ids = [str(p.id) for p in result]
        assert "a" in result_ids
        assert "b" in result_ids
        assert "c" in result_ids
        assert "d" in result_ids

    def test_rrf_scores_are_positive(self):
        """All fused scores should be positive."""
        points = [
            self._create_scored_point("a", 0.9),
            self._create_scored_point("b", 0.1),
        ]
        
        result = reciprocal_rank_fusion([points])
        
        for point in result:
            assert point.score > 0

    def test_k_parameter_affects_scores(self):
        """Different k values should produce different score distributions."""
        points = [
            self._create_scored_point("a", 0.9),
            self._create_scored_point("b", 0.8),
        ]
        
        result_k60 = reciprocal_rank_fusion([points], k=60)
        result_k10 = reciprocal_rank_fusion([points], k=10)
        
        # With smaller k, the difference between ranks is more pronounced
        score_diff_k60 = result_k60[0].score - result_k60[1].score
        score_diff_k10 = result_k10[0].score - result_k10[1].score
        
        # k=10 should have larger relative difference
        assert score_diff_k10 > score_diff_k60

    def test_payload_preserved(self):
        """Original payload should be preserved in fused results."""
        points = [
            self._create_scored_point("a", 0.9, "content for a"),
        ]
        
        result = reciprocal_rank_fusion([points])
        
        assert result[0].payload is not None
        assert result[0].payload["content"] == "content for a"

    def test_disjoint_lists(self):
        """RRF with completely different documents should include all."""
        list1 = [self._create_scored_point("a", 0.9)]
        list2 = [self._create_scored_point("b", 0.8)]
        
        result = reciprocal_rank_fusion([list1, list2])
        
        assert len(result) == 2
        result_ids = {str(p.id) for p in result}
        assert result_ids == {"a", "b"}
