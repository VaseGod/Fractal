"""
Fractal — Benchmark Runner
ARC-AGI-3 developer toolkit integration for evaluating
agent performance in zero-context environments.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

ARC_AGI_ENDPOINT = os.getenv("ARC_AGI_ENDPOINT", "https://api.arc-agi.org/v3")
ARC_AGI_API_KEY = os.getenv("ARC_AGI_API_KEY", "")


# ── Pydantic Models ──
class BenchmarkTask(BaseModel):
    """A single benchmark evaluation task."""

    task_id: str
    task_type: str  # "spatial_logic", "pattern_recognition", "sequence", etc.
    input_data: dict[str, Any]
    expected_output: dict[str, Any] | None = None
    max_steps: int = 50
    time_limit_seconds: int = 120


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark run."""

    benchmark_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = "arc_agi_3"
    num_tasks: int = Field(default=10, ge=1, le=1000)
    max_steps_per_task: int = Field(default=50, ge=1, le=500)
    time_limit_seconds: int = Field(default=120, ge=10, le=3600)
    zero_context: bool = True  # Deploy without prior prompting
    environment: str = "default"  # "default", "ls20_spatial", "custom"


class TaskScore(BaseModel):
    """Score for a single benchmark task."""

    task_id: str
    solved: bool = False
    actions_taken: int = 0
    information_gained: float = 0.0
    efficiency_ratio: float = 0.0  # actions / info_gained
    time_elapsed_seconds: float = 0.0
    error: str | None = None


class BenchmarkResult(BaseModel):
    """Complete results from a benchmark run."""

    benchmark_id: str
    task_type: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    total_tasks: int = 0
    tasks_solved: int = 0
    solve_rate: float = 0.0
    avg_efficiency_ratio: float = 0.0
    avg_actions_per_task: float = 0.0
    avg_time_per_task_seconds: float = 0.0
    total_time_seconds: float = 0.0
    task_scores: list[TaskScore] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRunner:
    """
    Benchmark runner integrating with the ARC-AGI-3 developer toolkit.

    Supports:
    - Loading evaluation tasks from ARC-AGI-3 API
    - Zero-context deployment (no prior prompting)
    - Efficiency scoring (actions/information gained)
    - Results export for Meta Agent feedback loop
    """

    def __init__(
        self,
        api_endpoint: str | None = None,
        api_key: str | None = None,
    ):
        self.api_endpoint = api_endpoint or ARC_AGI_ENDPOINT
        self.api_key = api_key or ARC_AGI_API_KEY
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=300.0,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def load_tasks(
        self, config: BenchmarkConfig
    ) -> list[BenchmarkTask]:
        """
        Load benchmark tasks from the ARC-AGI-3 API.
        Falls back to local sample tasks if API is unavailable.
        """
        logger.info(
            "benchmark.load_tasks",
            task_type=config.task_type,
            num_tasks=config.num_tasks,
        )

        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.api_endpoint}/tasks",
                params={
                    "type": config.task_type,
                    "count": config.num_tasks,
                    "environment": config.environment,
                },
            )
            response.raise_for_status()
            data = response.json()

            tasks = []
            for task_data in data.get("tasks", []):
                tasks.append(
                    BenchmarkTask(
                        task_id=task_data["id"],
                        task_type=config.task_type,
                        input_data=task_data["input"],
                        expected_output=task_data.get("expected_output"),
                        max_steps=config.max_steps_per_task,
                        time_limit_seconds=config.time_limit_seconds,
                    )
                )
            return tasks

        except Exception as e:
            logger.warning(
                "benchmark.api_unavailable",
                error=str(e),
                fallback="sample_tasks",
            )
            return self._generate_sample_tasks(config)

    def _generate_sample_tasks(
        self, config: BenchmarkConfig
    ) -> list[BenchmarkTask]:
        """Generate local sample tasks for testing."""
        tasks = []
        for i in range(config.num_tasks):
            task_id = f"sample_{config.task_type}_{i:04d}"

            if config.task_type == "spatial_logic":
                tasks.append(
                    BenchmarkTask(
                        task_id=task_id,
                        task_type="spatial_logic",
                        input_data={
                            "grid": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                            "instruction": "Identify the pattern and predict the next state.",
                        },
                        expected_output={"grid": [[1, 0, 1], [0, 1, 0], [1, 0, 1]]},
                        max_steps=config.max_steps_per_task,
                        time_limit_seconds=config.time_limit_seconds,
                    )
                )
            else:
                tasks.append(
                    BenchmarkTask(
                        task_id=task_id,
                        task_type=config.task_type,
                        input_data={
                            "sequence": [1, 2, 4, 8],
                            "instruction": "Predict the next 2 elements.",
                        },
                        expected_output={"sequence": [16, 32]},
                        max_steps=config.max_steps_per_task,
                        time_limit_seconds=config.time_limit_seconds,
                    )
                )

        return tasks

    async def run_benchmark(
        self, config: BenchmarkConfig
    ) -> BenchmarkResult:
        """
        Run a complete benchmark evaluation.

        For each task:
        1. Deploy agent in zero-context mode (if configured)
        2. Present the task input
        3. Track actions taken and information gained
        4. Score efficiency
        """
        logger.info(
            "benchmark.run_start",
            benchmark_id=config.benchmark_id,
            task_type=config.task_type,
        )

        tasks = await self.load_tasks(config)
        task_scores: list[TaskScore] = []
        total_start = datetime.now(timezone.utc)

        for task in tasks:
            score = await self._evaluate_task(task, config.zero_context)
            task_scores.append(score)

        # Compute aggregate metrics
        total_time = (
            datetime.now(timezone.utc) - total_start
        ).total_seconds()
        solved_count = sum(1 for s in task_scores if s.solved)
        total_tasks = len(task_scores)

        avg_efficiency = 0.0
        valid_efficiencies = [
            s.efficiency_ratio for s in task_scores if s.efficiency_ratio > 0
        ]
        if valid_efficiencies:
            avg_efficiency = sum(valid_efficiencies) / len(valid_efficiencies)

        avg_actions = 0.0
        if task_scores:
            avg_actions = sum(s.actions_taken for s in task_scores) / len(
                task_scores
            )

        avg_time = 0.0
        if task_scores:
            avg_time = sum(s.time_elapsed_seconds for s in task_scores) / len(
                task_scores
            )

        result = BenchmarkResult(
            benchmark_id=config.benchmark_id,
            task_type=config.task_type,
            total_tasks=total_tasks,
            tasks_solved=solved_count,
            solve_rate=solved_count / total_tasks if total_tasks > 0 else 0,
            avg_efficiency_ratio=avg_efficiency,
            avg_actions_per_task=avg_actions,
            avg_time_per_task_seconds=avg_time,
            total_time_seconds=total_time,
            task_scores=task_scores,
            metadata={
                "zero_context": config.zero_context,
                "environment": config.environment,
                "max_steps_per_task": config.max_steps_per_task,
            },
        )

        logger.info(
            "benchmark.run_complete",
            benchmark_id=config.benchmark_id,
            solve_rate=result.solve_rate,
            avg_efficiency=avg_efficiency,
        )

        return result

    async def _evaluate_task(
        self, task: BenchmarkTask, zero_context: bool
    ) -> TaskScore:
        """
        Evaluate a single benchmark task.
        In production, this delegates to the Task Agent.
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Import Task Agent for evaluation
            from src.agents.task_agent import TaskAgent, TaskInput

            agent = TaskAgent()

            # Build task input
            task_input = TaskInput(
                objective=(
                    f"Solve this {task.task_type} task: "
                    f"{json.dumps(task.input_data)}"
                ),
                context={
                    "benchmark_task_id": task.task_id,
                    "zero_context": str(zero_context),
                },
                max_steps=task.max_steps,
            )

            # Execute
            result = await agent.execute(task_input)

            elapsed = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds()

            # Determine if solved (compare output to expected)
            solved = False
            if task.expected_output and result.status == "success":
                solved = self._check_solution(
                    result.result, task.expected_output
                )

            # Calculate efficiency
            actions = result.steps_taken
            info_gained = 1.0 if solved else 0.5  # Simplified metric
            efficiency = actions / info_gained if info_gained > 0 else float("inf")

            return TaskScore(
                task_id=task.task_id,
                solved=solved,
                actions_taken=actions,
                information_gained=info_gained,
                efficiency_ratio=efficiency,
                time_elapsed_seconds=elapsed,
            )

        except Exception as e:
            elapsed = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds()
            logger.error(
                "benchmark.task_error",
                task_id=task.task_id,
                error=str(e),
            )
            return TaskScore(
                task_id=task.task_id,
                solved=False,
                time_elapsed_seconds=elapsed,
                error=str(e),
            )

    @staticmethod
    def _check_solution(
        actual: dict[str, Any], expected: dict[str, Any]
    ) -> bool:
        """Compare agent output to expected solution."""
        try:
            return json.dumps(actual, sort_keys=True) == json.dumps(
                expected, sort_keys=True
            )
        except (TypeError, ValueError):
            return False

    async def export_results(
        self, result: BenchmarkResult, output_path: str | None = None
    ) -> str:
        """Export benchmark results to a JSON file."""
        output_path = output_path or os.path.join(
            os.getenv("FRACTAL_DATA_DIR", "/app/data"),
            "benchmarks",
            f"{result.benchmark_id}.json",
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)

        logger.info("benchmark.exported", path=output_path)
        return output_path
