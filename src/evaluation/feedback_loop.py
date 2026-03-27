"""
Fractal — Feedback Loop
Wires benchmark results back into the Meta Agent
to refine system configuration and prompt strategies.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

from src.evaluation.benchmark import BenchmarkResult, BenchmarkRunner, BenchmarkConfig
from src.evaluation.scoring import EfficiencyScorer, EfficiencyMetrics, BatchEfficiencyReport

logger = structlog.get_logger(__name__)

DATA_DIR = Path(os.getenv("FRACTAL_DATA_DIR", "/app/data"))


class FeedbackCycleConfig(BaseModel):
    """Configuration for a feedback loop cycle."""

    cycle_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    benchmark_type: str = "arc_agi_3"
    num_benchmark_tasks: int = 10
    auto_archive: bool = True
    auto_propose: bool = True
    improvement_threshold: float = 0.01  # Min score delta to consider improvement


class FeedbackCycleResult(BaseModel):
    """Result from a complete feedback cycle."""

    cycle_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    benchmark_result: BenchmarkResult | None = None
    efficiency_report: BatchEfficiencyReport | None = None
    comparison: dict[str, Any] | None = None
    meta_analysis_triggered: bool = False
    config_archived: bool = False
    proposals_generated: int = 0
    status: str = "completed"
    error: str | None = None


class FeedbackLoop:
    """
    Feedback loop connecting benchmarks to the Meta Agent.

    Complete cycle:
    1. Run benchmark suite
    2. Score efficiency
    3. Compare against baseline
    4. If improved: archive configuration
    5. Feed results to Meta Agent for analysis
    6. Meta Agent proposes optimizations
    7. Apply approved optimizations
    8. Repeat
    """

    def __init__(self):
        self.benchmark_runner = BenchmarkRunner()
        self.scorer = EfficiencyScorer()
        self._baseline: BatchEfficiencyReport | None = None

        # Load previous baseline if available
        baseline_path = DATA_DIR / "benchmarks" / "baseline.json"
        if baseline_path.exists():
            try:
                with open(baseline_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._baseline = BatchEfficiencyReport(**data)
                logger.info("feedback_loop.baseline_loaded")
            except Exception as e:
                logger.warning("feedback_loop.baseline_load_error", error=str(e))

    async def run_cycle(
        self, config: FeedbackCycleConfig
    ) -> FeedbackCycleResult:
        """
        Run a complete feedback cycle:
        benchmark → score → compare → meta-analyze → optimize
        """
        logger.info(
            "feedback_loop.cycle_start",
            cycle_id=config.cycle_id,
            benchmark_type=config.benchmark_type,
        )

        result = FeedbackCycleResult(cycle_id=config.cycle_id)

        try:
            # Step 1: Run benchmark
            benchmark_config = BenchmarkConfig(
                task_type=config.benchmark_type,
                num_tasks=config.num_benchmark_tasks,
                zero_context=True,
            )
            benchmark_result = await self.benchmark_runner.run_benchmark(
                benchmark_config
            )
            result.benchmark_result = benchmark_result

            # Step 2: Score efficiency
            task_metrics = []
            for task_score in benchmark_result.task_scores:
                metric = self.scorer.score_task(
                    actions_taken=task_score.actions_taken,
                    task_solved=task_score.solved,
                    time_elapsed_seconds=task_score.time_elapsed_seconds,
                    novel_observations=1 if task_score.solved else 0,
                )
                task_metrics.append(metric)

            efficiency_report = self.scorer.score_batch(task_metrics)
            result.efficiency_report = efficiency_report

            # Step 3: Compare against baseline
            if self._baseline:
                comparison = self.scorer.compare_runs(
                    self._baseline, efficiency_report
                )
                result.comparison = comparison

                improved = comparison.get("improved", False)

                if improved and config.auto_archive:
                    # Step 4: Archive configuration
                    await self._archive_current_config(
                        benchmark_result, efficiency_report
                    )
                    result.config_archived = True

                    # Update baseline
                    self._baseline = efficiency_report
                    await self._save_baseline(efficiency_report)

            else:
                # First run — set as baseline
                self._baseline = efficiency_report
                await self._save_baseline(efficiency_report)
                result.comparison = {"note": "First run — set as baseline."}

            # Step 5: Feed to Meta Agent
            if config.auto_propose:
                proposals = await self._trigger_meta_analysis(
                    benchmark_result, efficiency_report
                )
                result.meta_analysis_triggered = True
                result.proposals_generated = proposals

            # Save cycle results
            await self._save_cycle_result(result)

            logger.info(
                "feedback_loop.cycle_complete",
                cycle_id=config.cycle_id,
                solve_rate=benchmark_result.solve_rate,
                avg_efficiency=efficiency_report.avg_normalized_score,
            )

        except Exception as e:
            result.status = "error"
            result.error = str(e)
            logger.error(
                "feedback_loop.cycle_error",
                cycle_id=config.cycle_id,
                error=str(e),
            )

        return result

    async def _archive_current_config(
        self,
        benchmark_result: BenchmarkResult,
        efficiency_report: BatchEfficiencyReport,
    ) -> None:
        """Archive the current agent configuration with its benchmark scores."""
        from src.agents.tools.eval_tools import archive_configuration

        await archive_configuration.ainvoke(
            {
                "config_name": f"benchmark_{benchmark_result.benchmark_id[:8]}",
                "config_data": {
                    "benchmark_id": benchmark_result.benchmark_id,
                    "task_type": benchmark_result.task_type,
                    "solve_rate": benchmark_result.solve_rate,
                    "model": os.getenv("VLLM_MODEL_PATH", "default"),
                },
                "performance_metrics": {
                    "solve_rate": benchmark_result.solve_rate,
                    "avg_efficiency": efficiency_report.avg_normalized_score,
                    "avg_actions": benchmark_result.avg_actions_per_task,
                },
                "tags": ["benchmark", benchmark_result.task_type, "auto"],
            }
        )

    async def _trigger_meta_analysis(
        self,
        benchmark_result: BenchmarkResult,
        efficiency_report: BatchEfficiencyReport,
    ) -> int:
        """Trigger Meta Agent analysis with benchmark results."""
        try:
            from src.agents.meta_agent import MetaAgent, MetaAnalysisInput

            meta = MetaAgent()
            analysis = await meta.run_analysis_cycle(
                MetaAnalysisInput(
                    time_window_hours=1,
                    include_proposals=True,
                    max_proposals=5,
                )
            )
            return len(analysis.proposals)

        except Exception as e:
            logger.warning(
                "feedback_loop.meta_analysis_error", error=str(e)
            )
            return 0

    async def _save_baseline(
        self, report: BatchEfficiencyReport
    ) -> None:
        """Save the current baseline efficiency report."""
        baseline_dir = DATA_DIR / "benchmarks"
        baseline_dir.mkdir(parents=True, exist_ok=True)

        baseline_path = baseline_dir / "baseline.json"
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2, default=str)

    async def _save_cycle_result(
        self, result: FeedbackCycleResult
    ) -> None:
        """Save feedback cycle result for audit trail."""
        cycles_dir = DATA_DIR / "feedback_cycles"
        cycles_dir.mkdir(parents=True, exist_ok=True)

        result_path = cycles_dir / f"{result.cycle_id}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)


# ── CLI Entry Point ──
if __name__ == "__main__":
    import asyncio

    async def main():
        loop = FeedbackLoop()
        result = await loop.run_cycle(FeedbackCycleConfig())
        print(json.dumps(result.model_dump(), indent=2, default=str))

    asyncio.run(main())
