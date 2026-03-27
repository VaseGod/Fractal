"""
Fractal — Efficiency Scoring
Calculates the ratio of actions taken to information gained
based on ARC-AGI-3 methodology.
"""

from __future__ import annotations

import math
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class EfficiencyMetrics(BaseModel):
    """Computed efficiency metrics for a task or batch."""

    raw_efficiency: float = 0.0  # actions / info_gained (lower = better)
    normalized_score: float = 0.0  # 0 to 1 (higher = better)
    action_count: int = 0
    information_gained: float = 0.0
    time_efficiency: float = 0.0  # info_gained / time_seconds
    step_quality: float = 0.0  # fraction of steps that advanced the task


class BatchEfficiencyReport(BaseModel):
    """Aggregated efficiency report across multiple tasks."""

    total_tasks: int = 0
    avg_normalized_score: float = 0.0
    median_normalized_score: float = 0.0
    min_normalized_score: float = 0.0
    max_normalized_score: float = 0.0
    std_dev_score: float = 0.0
    total_actions: int = 0
    total_information: float = 0.0
    global_efficiency: float = 0.0
    per_task_metrics: list[EfficiencyMetrics] = Field(default_factory=list)


class EfficiencyScorer:
    """
    Calculate efficiency scores using ARC-AGI-3 methodology.

    Core formula:
        efficiency = actions_taken / information_gained

    Normalized to a 0-1 scale where:
        - 1.0 = maximally efficient (minimum actions, maximum information)
        - 0.0 = maximally inefficient (many actions, no information)

    Information gained is estimated from:
        - Task completion status
        - Intermediate state changes
        - Novel observations made
    """

    def __init__(
        self,
        max_actions_baseline: int = 50,
        information_decay_rate: float = 0.95,
    ):
        self.max_actions_baseline = max_actions_baseline
        self.information_decay_rate = information_decay_rate

    def score_task(
        self,
        actions_taken: int,
        task_solved: bool,
        intermediate_states: list[dict[str, Any]] | None = None,
        time_elapsed_seconds: float = 0.0,
        novel_observations: int = 0,
    ) -> EfficiencyMetrics:
        """
        Score efficiency for a single task.

        Args:
            actions_taken: Total actions/tool calls made
            task_solved: Whether the task was successfully completed
            intermediate_states: List of state snapshots during execution
            time_elapsed_seconds: Total execution time
            novel_observations: Count of genuinely new information discovered
        """
        intermediate_states = intermediate_states or []

        # Calculate information gained
        info_gained = self._estimate_information_gained(
            task_solved=task_solved,
            intermediate_states=intermediate_states,
            novel_observations=novel_observations,
        )

        # Raw efficiency (lower is better)
        raw_efficiency = (
            actions_taken / info_gained if info_gained > 0 else float("inf")
        )

        # Normalized score (higher is better, 0-1 range)
        normalized = self._normalize_score(
            actions_taken=actions_taken,
            info_gained=info_gained,
            task_solved=task_solved,
        )

        # Step quality: fraction of steps that contributed information
        step_quality = 0.0
        if actions_taken > 0:
            productive_steps = min(novel_observations + len(intermediate_states), actions_taken)
            step_quality = productive_steps / actions_taken

        # Time efficiency
        time_efficiency = (
            info_gained / time_elapsed_seconds
            if time_elapsed_seconds > 0
            else 0.0
        )

        return EfficiencyMetrics(
            raw_efficiency=raw_efficiency,
            normalized_score=normalized,
            action_count=actions_taken,
            information_gained=info_gained,
            time_efficiency=time_efficiency,
            step_quality=step_quality,
        )

    def score_batch(
        self, task_scores: list[EfficiencyMetrics]
    ) -> BatchEfficiencyReport:
        """
        Compute aggregate efficiency across a batch of tasks.
        """
        if not task_scores:
            return BatchEfficiencyReport()

        scores = [s.normalized_score for s in task_scores]
        n = len(scores)

        avg_score = sum(scores) / n
        sorted_scores = sorted(scores)
        median_score = sorted_scores[n // 2]

        # Standard deviation
        variance = sum((s - avg_score) ** 2 for s in scores) / n
        std_dev = math.sqrt(variance)

        total_actions = sum(s.action_count for s in task_scores)
        total_info = sum(s.information_gained for s in task_scores)

        return BatchEfficiencyReport(
            total_tasks=n,
            avg_normalized_score=avg_score,
            median_normalized_score=median_score,
            min_normalized_score=min(scores),
            max_normalized_score=max(scores),
            std_dev_score=std_dev,
            total_actions=total_actions,
            total_information=total_info,
            global_efficiency=(
                total_actions / total_info if total_info > 0 else float("inf")
            ),
            per_task_metrics=task_scores,
        )

    def _estimate_information_gained(
        self,
        task_solved: bool,
        intermediate_states: list[dict[str, Any]],
        novel_observations: int,
    ) -> float:
        """
        Estimate information gained during task execution.

        Components:
        1. Base: task completion (1.0 if solved, 0.2 if not)
        2. State progression: each unique state transition adds info
        3. Novel observations: each new observation adds diminishing info
        """
        # Base information from task completion
        info = 1.0 if task_solved else 0.2

        # State progression (with diminishing returns)
        for i, _state in enumerate(intermediate_states):
            info += self.information_decay_rate ** i * 0.1

        # Novel observations (with diminishing returns)
        for i in range(novel_observations):
            info += self.information_decay_rate ** i * 0.15

        return info

    def _normalize_score(
        self,
        actions_taken: int,
        info_gained: float,
        task_solved: bool,
    ) -> float:
        """
        Normalize efficiency to a 0-1 scale.

        Uses a sigmoid-like function that rewards:
        - Fewer actions for the same information
        - Solving the task (significant bonus)
        """
        if actions_taken == 0:
            return 1.0 if task_solved else 0.0

        if info_gained <= 0:
            return 0.0

        # Ratio of optimal actions to actual actions
        optimal_actions = max(1, int(info_gained * 2))  # Estimate
        action_ratio = optimal_actions / actions_taken

        # Sigmoid normalization
        score = 1.0 / (1.0 + math.exp(-5 * (action_ratio - 0.5)))

        # Bonus for solving
        if task_solved:
            score = min(1.0, score * 1.2)

        return round(max(0.0, min(1.0, score)), 4)

    def compare_runs(
        self,
        baseline: BatchEfficiencyReport,
        current: BatchEfficiencyReport,
    ) -> dict[str, Any]:
        """
        Compare two benchmark runs to measure improvement.
        Returns delta metrics and a recommendation.
        """
        score_delta = (
            current.avg_normalized_score - baseline.avg_normalized_score
        )
        efficiency_delta = 0.0
        if baseline.global_efficiency > 0 and current.global_efficiency > 0:
            efficiency_delta = (
                baseline.global_efficiency - current.global_efficiency
            ) / baseline.global_efficiency

        improved = score_delta > 0.01  # Minimum meaningful improvement

        return {
            "score_delta": round(score_delta, 4),
            "efficiency_delta_pct": round(efficiency_delta * 100, 2),
            "baseline_avg_score": baseline.avg_normalized_score,
            "current_avg_score": current.avg_normalized_score,
            "improved": improved,
            "recommendation": (
                "Performance improved. Consider archiving current configuration."
                if improved
                else "No significant improvement. Review optimization proposals."
            ),
        }
