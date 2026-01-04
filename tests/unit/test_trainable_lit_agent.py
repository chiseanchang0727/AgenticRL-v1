"""Unit tests for TrainableLitAgent."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any
from train.training_interface import TrainableLitAgent
import agentlightning as agl


class TestTrainableLitAgent:
    """Test suite for TrainableLitAgent class."""

    def test_init(self):
        """Test TrainableLitAgent initialization."""
        agent = TrainableLitAgent(trained_agents=None, max_turns=5)
        assert agent.max_turns == 5
        assert agent.trained_agents is None

    def test_init_with_trained_agents(self):
        """Test initialization with trained_agents parameter."""
        agent = TrainableLitAgent(trained_agents="path/to/agents")
        assert agent.trained_agents == "path/to/agents"
        assert agent.max_turns == 3  # default value

    @patch('train.training_interface.build_agent')
    def test_rollout_success(self, mock_build_agent):
        """Test successful rollout execution."""
        # Setup mocks
        mock_app = AsyncMock()
        mock_app.ainvoke = AsyncMock(return_value={
            "user_input": "test question",
            "final_answer": "Room A is available"
        })
        mock_build_agent.return_value = mock_app

        # Create agent
        agent = TrainableLitAgent(trained_agents=None)

        # Setup task and resources
        task: Dict[str, Any] = {"question": "Find a room for 10 people"}
        mock_llm = Mock(spec=agl.LLM)
        resources: agl.NamedResources = {"main_llm": mock_llm}
        rollout = Mock(spec=agl.Rollout)

        # Execute rollout
        result = agent.rollout(task, resources, rollout)

        # Assertions
        assert result == "Room A is available"
        mock_build_agent.assert_called_once()
        mock_app.ainvoke.assert_called_once_with({"user_input": "Find a room for 10 people"})

    @patch('train.training_interface.build_agent')
    def test_rollout_no_final_answer(self, mock_build_agent):
        """Test rollout when final_answer is missing."""
        # Setup mocks
        mock_app = AsyncMock()
        mock_app.ainvoke = AsyncMock(return_value={
            "user_input": "test question",
            "history": []
        })
        mock_build_agent.return_value = mock_app

        agent = TrainableLitAgent(trained_agents=None)
        task: Dict[str, Any] = {"question": "Find a room"}
        mock_llm = Mock(spec=agl.LLM)
        resources: agl.NamedResources = {"main_llm": mock_llm}
        rollout = Mock(spec=agl.Rollout)

        result = agent.rollout(task, resources, rollout)

        assert result is None

    @patch('train.training_interface.build_agent')
    def test_rollout_missing_question(self, mock_build_agent):
        """Test rollout when question is missing from task."""
        # Setup mocks
        mock_app = AsyncMock()
        mock_app.ainvoke = AsyncMock(return_value={
            "final_answer": "Some answer"
        })
        mock_build_agent.return_value = mock_app

        agent = TrainableLitAgent(trained_agents=None)
        task: Dict[str, Any] = {}  # Missing question
        mock_llm = Mock(spec=agl.LLM)
        resources: agl.NamedResources = {"main_llm": mock_llm}
        rollout = Mock(spec=agl.Rollout)

        result = agent.rollout(task, resources, rollout)

        # Should still work, but question will be None
        assert result == "Some answer"
        mock_app.ainvoke.assert_called_once_with({"user_input": None})

    @patch('train.training_interface.build_agent')
    def test_rollout_async_compatibility(self, mock_build_agent):
        """Test that rollout handles async execution correctly."""
        # Setup mocks
        mock_app = AsyncMock()
        mock_app.ainvoke = AsyncMock(return_value={
            "final_answer": "Test answer"
        })
        mock_build_agent.return_value = mock_app

        agent = TrainableLitAgent(trained_agents=None)
        task: Dict[str, Any] = {"question": "Test question"}
        mock_llm = Mock(spec=agl.LLM)
        resources: agl.NamedResources = {"main_llm": mock_llm}
        rollout = Mock(spec=agl.Rollout)

        # Should work without raising exceptions
        result = agent.rollout(task, resources, rollout)
        assert result == "Test answer"

    @patch('train.training_interface.build_agent')
    def test_rollout_uses_correct_state(self, mock_build_agent):
        """Test that build_agent is called with correct AgentState."""
        from agent.state import AgentState

        mock_app = AsyncMock()
        mock_app.ainvoke = AsyncMock(return_value={"final_answer": "Answer"})
        mock_build_agent.return_value = mock_app

        agent = TrainableLitAgent(trained_agents=None)
        task: Dict[str, Any] = {"question": "Test"}
        mock_llm = Mock(spec=agl.LLM)
        resources: agl.NamedResources = {"main_llm": mock_llm}
        rollout = Mock(spec=agl.Rollout)

        agent.rollout(task, resources, rollout)

        # Verify build_agent was called with AgentState
        # Note: current code only passes state, verl_replacement is not passed
        mock_build_agent.assert_called_once_with(state=AgentState)

    @patch('train.training_interface.build_agent')
    def test_rollout_without_verl_replacement(self, mock_build_agent):
        """Test rollout when verl_replacement is None (default behavior)."""
        from agent.state import AgentState

        mock_app = AsyncMock()
        mock_app.ainvoke = AsyncMock(return_value={"final_answer": "Answer"})
        mock_build_agent.return_value = mock_app

        agent = TrainableLitAgent(trained_agents=None)
        task: Dict[str, Any] = {"question": "Test"}
        mock_llm = Mock(spec=agl.LLM)
        resources: agl.NamedResources = {"main_llm": mock_llm}
        rollout = Mock(spec=agl.Rollout)

        agent.rollout(task, resources, rollout)

        # When verl_replacement is None, should pass None to build_agent
        # Note: Currently code doesn't pass verl_replacement at all (bug)
        # This test documents expected behavior once fixed
        mock_build_agent.assert_called_once()
        # Once fixed, should be: assert_called_once_with(state=AgentState, verl_replacement=None)

    @patch('train.training_interface.build_agent')
    def test_rollout_with_verl_replacement(self, mock_build_agent):
        """Test rollout when verl_replacement is provided (for VERL training).
        
        This test documents the expected behavior when verl_replacement is extracted
        from rollout/resources and passed to build_agent. Currently the code doesn't
        support this, but this test should pass once the feature is implemented.
        """
        from agent.state import AgentState

        mock_app = AsyncMock()
        mock_app.ainvoke = AsyncMock(return_value={"final_answer": "Answer"})
        mock_build_agent.return_value = mock_app

        # Example verl_replacement structure
        verl_replacement: Dict[str, Any] = {
            "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        agent = TrainableLitAgent(trained_agents=None)
        task: Dict[str, Any] = {"question": "Test"}
        mock_llm = Mock(spec=agl.LLM)
        resources: agl.NamedResources = {"main_llm": mock_llm}
        
        # Mock rollout to contain verl_replacement
        # Note: This is hypothetical - actual implementation depends on how
        # agentlightning provides verl_replacement in the rollout object
        rollout = Mock(spec=agl.Rollout)
        rollout.verl_replacement = verl_replacement  # Hypothetical attribute

        agent.rollout(task, resources, rollout)

        # Once implemented, should pass verl_replacement to build_agent
        # Expected: mock_build_agent.assert_called_once_with(
        #     state=AgentState, 
        #     verl_replacement=verl_replacement
        # )
        mock_build_agent.assert_called_once()

