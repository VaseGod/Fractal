"""
Fractal — Benchmark Tests
Tests for benchmark runner and scoring.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.evaluation.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkTask,
    TaskScore,
)


class TestBenchmarkConfig:
    """Test BenchmarkConfig validation."""

    def test_defaults(self):
        cfg = BenchmarkConfig()
        assert cfg.task_type == "arc_agi_3"
        assert cfg.num_tasks == 10
        assert cfg.zero_context is True
        assert cfg.max_steps_per_task == 50
        assert len(cfg.benchmark_id) > 0

    def test_custom(self):
        cfg = BenchmarkConfig(
            task_type="spatial_logic",
            num_tasks=50,
            environment="ls20_spatial",
        )
        assert cfg.task_type == "spatial_logic"
        assert cfg.num_tasks == 50
        assert cfg.environment == "ls20_spatial"

    def test_bounds(self):
        with pytest.raises(ValidationError):
            BenchmarkConfig(num_tasks=0)

        with pytest.raises(ValidationError):
            BenchmarkConfig(num_tasks=1001)


class TestBenchmarkTask:
    """Test BenchmarkTask model."""

    def test_creation(self):
        task = BenchmarkTask(
            task_id="task-001",
            task_type="spatial_logic",
            input_data={"grid": [[0, 1], [1, 0]]},
            expected_output={"grid": [[1, 0], [0, 1]]},
        )
        assert task.task_id == "task-001"
        assert task.max_steps == 50


class TestTaskScore:
    """Test TaskScore model."""

    def test_solved(self):
        score = TaskScore(
            task_id="task-001",
            solved=True,
            actions_taken=5,
            information_gained=1.0,
            efficiency_ratio=5.0,
            time_elapsed_seconds=2.5,
        )
        assert score.solved is True
        assert score.efficiency_ratio == 5.0

    def test_unsolved(self):
        score = TaskScore(
            task_id="task-002",
            solved=False,
            actions_taken=50,
            error="TimeoutError",
        )
        assert score.solved is False
        assert score.error == "TimeoutError"


class TestBenchmarkResult:
    """Test BenchmarkResult model."""

    def test_empty(self):
        result = BenchmarkResult(
            benchmark_id="bench-001",
            task_type="arc_agi_3",
        )
        assert result.total_tasks == 0
        assert result.solve_rate == 0.0

    def test_populated(self):
        result = BenchmarkResult(
            benchmark_id="bench-002",
            task_type="spatial_logic",
            total_tasks=10,
            tasks_solved=7,
            solve_rate=0.7,
            avg_efficiency_ratio=3.5,
            task_scores=[
                TaskScore(
                    task_id=f"task-{i}",
                    solved=i < 7,
                    actions_taken=i + 1,
                )
                for i in range(10)
            ],
        )
        assert result.solve_rate == 0.7
        assert len(result.task_scores) == 10
