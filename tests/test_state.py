"""
Tests for the AgentState module.
"""

import pytest
from langchain_core.documents import Document

from src.state.agent_state import AgentState, create_initial_state


class TestAgentState:
    """Test suite for AgentState."""

    def test_create_initial_state(self):
        """create_initial_state should return properly initialized state."""
        state = create_initial_state("What is LangGraph?")
        
        assert state["question"] == "What is LangGraph?"
        assert state["documents"] == []
        assert state["generation"] == ""
        assert state["web_search"] == "No"
        assert state["grading_status"] == "pending"
        assert state["query_rewrite_count"] == 0

    def test_state_is_mutable(self):
        """AgentState should support mutation."""
        state = create_initial_state("test")
        
        # Add documents
        doc = Document(page_content="test content")
        state["documents"] = [doc]
        
        assert len(state["documents"]) == 1
        assert state["documents"][0].page_content == "test content"

    def test_state_update_pattern(self):
        """State update pattern used by nodes should work correctly."""
        initial = create_initial_state("test")
        
        # Simulate node update pattern
        updated = {
            **initial,
            "generation": "This is the answer",
            "grading_status": "relevant",
        }
        
        assert updated["question"] == "test"
        assert updated["generation"] == "This is the answer"
        assert updated["grading_status"] == "relevant"
        # Unchanged fields
        assert updated["web_search"] == "No"
        assert updated["query_rewrite_count"] == 0
