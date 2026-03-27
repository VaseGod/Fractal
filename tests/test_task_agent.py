"""
Fractal — Task Agent Tests
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
from pydantic import ValidationError

from src.agents.task_agent import TaskInput, TaskResult


class TestTaskInput:
    """Test TaskInput Pydantic model validation."""

    def test_valid_input(self):
        task = TaskInput(objective="Test task objective")
        assert task.objective == "Test task objective"
        assert task.max_steps == 25
        assert task.priority == "normal"
        assert task.require_hitl is False
        assert len(task.task_id) > 0

    def test_priority_validation(self):
        task = TaskInput(objective="Test", priority="critical")
        assert task.priority == "critical"

    def test_invalid_priority_rejected(self):
        with pytest.raises(ValidationError):
            TaskInput(objective="Test", priority="invalid_priority")

    def test_empty_objective_rejected(self):
        with pytest.raises(ValidationError):
            TaskInput(objective="")

    def test_max_steps_bounds(self):
        task = TaskInput(objective="Test", max_steps=1)
        assert task.max_steps == 1

        task = TaskInput(objective="Test", max_steps=100)
        assert task.max_steps == 100

        with pytest.raises(ValidationError):
            TaskInput(objective="Test", max_steps=0)

        with pytest.raises(ValidationError):
            TaskInput(objective="Test", max_steps=101)

    def test_context_default(self):
        task = TaskInput(objective="Test")
        assert task.context == {}

    def test_context_with_data(self):
        task = TaskInput(
            objective="Test", context={"key": "value", "nested": {"a": 1}}
        )
        assert task.context["key"] == "value"


class TestTaskResult:
    """Test TaskResult Pydantic model validation."""

    def test_success_result(self):
        result = TaskResult(
            task_id="test-123",
            status="success",
            result={"output": "done"},
            steps_taken=5,
            confidence=0.9,
        )
        assert result.status == "success"
        assert result.confidence == 0.9
        assert result.steps_taken == 5

    def test_failure_result(self):
        result = TaskResult(
            task_id="test-456",
            status="failure",
            errors=["TimeoutError: inference timed out"],
        )
        assert result.status == "failure"
        assert len(result.errors) == 1

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            TaskResult(task_id="test", status="invalid_status")

    def test_confidence_bounds(self):
        result = TaskResult(task_id="test", status="success", confidence=1.0)
        assert result.confidence == 1.0

        with pytest.raises(ValidationError):
            TaskResult(task_id="test", status="success", confidence=1.5)

        with pytest.raises(ValidationError):
            TaskResult(task_id="test", status="success", confidence=-0.1)

    def test_timestamp_auto_set(self):
        result = TaskResult(task_id="test", status="success")
        assert result.timestamp is not None
        assert len(result.timestamp) > 0
