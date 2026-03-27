"""
Fractal — Meta Agent
Metacognitive evaluator that analyzes Task Agent performance,
identifies optimization opportunities, and proposes system improvements.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langsmith import Client as LangSmithClient
from langsmith import traceable
from pydantic import BaseModel, Field

from src.agents.tools.eval_tools import get_eval_tools
from src.agents.tools.memory_tools import get_memory_tools
from src.middleware.hitl_gate import HITLGate

load_dotenv()
logger = structlog.get_logger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_PORT = os.getenv("VLLM_PORT", "8000")
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
DATA_DIR = Path(os.getenv("FRACTAL_DATA_DIR", "/app/data"))


# ── Pydantic Models ──
class MetaAnalysisInput(BaseModel):
    """Input for a meta-analysis cycle."""

    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    time_window_hours: int = Field(default=24, ge=1, le=720)
    metrics_to_compute: list[str] = Field(
        default=["error_rate", "completion_time", "tool_efficiency", "hitl_rate"]
    )
    include_proposals: bool = Field(default=True)
    max_proposals: int = Field(default=5, ge=1, le=20)


class PerformanceMetrics(BaseModel):
    """Computed performance metrics."""

    error_rate: float = 0.0
    avg_completion_time_ms: float = 0.0
    p95_completion_time_ms: float = 0.0
    tool_efficiency: float = 0.0
    hitl_trigger_rate: float = 0.0
    hitl_approval_rate: float = 0.0
    total_executions: int = 0


class FailurePattern(BaseModel):
    """An identified failure pattern."""

    pattern_type: str  # "failure", "success", "drift"
    description: str
    evidence: list[str] = Field(default_factory=list)
    frequency: float = 0.0
    confidence: float = 0.0


class OptimizationProposal(BaseModel):
    """A proposed system optimization."""

    proposal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action: str  # "modify_eval", "modify_config", "archive_strategy", "adjust_prompt"
    target: str
    change_description: str
    expected_impact: str
    rollback_procedure: str
    requires_hitl: bool = True
    confidence: float = 0.0


class MetaAnalysisResult(BaseModel):
    """Full result from a meta-analysis cycle."""

    analysis_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metrics: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    patterns: list[FailurePattern] = Field(default_factory=list)
    proposals: list[OptimizationProposal] = Field(default_factory=list)
    memory_entries_stored: int = 0
    configs_archived: int = 0


class MetaAgent:
    """
    Metacognitive evaluator agent.

    Runs alongside the Task Agent to:
    1. Ingest execution logs and LangSmith traces
    2. Compute performance metrics (error rates, latency, efficiency)
    3. Identify failure patterns and success strategies
    4. Propose optimization changes (with HITL approval)
    5. Archive winning configurations in vector memory
    """

    def __init__(
        self,
        model_name: str = "default",
        temperature: float = 0.2,
    ):
        self.agent_id = str(uuid.uuid4())
        self.hitl_gate = HITLGate()

        # Load meta system prompt
        system_prompt_path = PROMPTS_DIR / "meta_system.txt"
        self.system_prompt = system_prompt_path.read_text(encoding="utf-8")

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key="EMPTY",
            openai_api_base=VLLM_BASE_URL,
            temperature=temperature,
            max_tokens=4096,
        )

        # Meta Agent has access to eval and memory tools (NOT web tools)
        self.tools = [
            *get_eval_tools(),
            *get_memory_tools(),
        ]

        # Create the react agent graph using langgraph
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=self.system_prompt,
        )

        # LangSmith client for trace ingestion
        self.langsmith_client: LangSmithClient | None = None
        api_key = os.getenv("LANGCHAIN_API_KEY")
        if api_key and api_key != "ls_your_api_key_here":
            try:
                self.langsmith_client = LangSmithClient(api_key=api_key)
            except Exception as e:
                logger.warning("meta_agent.langsmith_init_failed", error=str(e))

        logger.info(
            "meta_agent.initialized",
            agent_id=self.agent_id,
            tool_count=len(self.tools),
        )

    @traceable(name="meta_agent.analyze", run_type="chain")
    async def run_analysis_cycle(
        self, analysis_input: MetaAnalysisInput
    ) -> MetaAnalysisResult:
        """
        Run a full metacognitive analysis cycle.

        1. Fetch Task Agent execution traces from LangSmith
        2. Compute performance metrics
        3. Use the LLM to identify patterns and propose optimizations
        4. Archive successful strategies in vector memory
        """
        logger.info(
            "meta_agent.analysis_start",
            analysis_id=analysis_input.analysis_id,
            time_window_hours=analysis_input.time_window_hours,
        )

        # Step 1: Gather execution data
        traces = await self._fetch_traces(analysis_input.time_window_hours)

        # Step 2: Compute metrics
        metrics = self._compute_metrics(traces)

        # Step 3: Use the agent to analyze patterns and propose changes
        analysis_prompt = self._build_analysis_prompt(
            metrics, traces, analysis_input
        )

        try:
            result = await self.agent.ainvoke(
                {"messages": [{"role": "user", "content": analysis_prompt}]},
            )

            # Step 4: Parse agent output for patterns and proposals
            messages = result.get("messages", [])
            output_text = str(messages[-1].content) if messages else ""
            patterns, proposals = self._parse_agent_output(
                output_text
            )

            # Step 5: Store successful patterns in memory
            memory_count = await self._store_patterns_in_memory(patterns)

            # Step 6: Archive current configuration if metrics improved
            configs_archived = await self._maybe_archive_config(metrics)

            return MetaAnalysisResult(
                analysis_id=analysis_input.analysis_id,
                metrics=metrics,
                patterns=patterns,
                proposals=proposals[: analysis_input.max_proposals],
                memory_entries_stored=memory_count,
                configs_archived=configs_archived,
            )

        except Exception as e:
            logger.error(
                "meta_agent.analysis_error",
                analysis_id=analysis_input.analysis_id,
                error=str(e),
            )
            return MetaAnalysisResult(
                analysis_id=analysis_input.analysis_id,
                metrics=metrics,
            )

    async def _fetch_traces(
        self, time_window_hours: int
    ) -> list[dict[str, Any]]:
        """Fetch execution traces from LangSmith."""
        traces: list[dict[str, Any]] = []

        if self.langsmith_client:
            try:
                project_name = os.getenv(
                    "LANGCHAIN_PROJECT", "fractal-production"
                )
                runs = self.langsmith_client.list_runs(
                    project_name=project_name,
                    run_type="chain",
                    filter=f'gte(start_time, "{time_window_hours}h")',
                )
                for run in runs:
                    traces.append(
                        {
                            "trace_id": str(run.id),
                            "status": run.status,
                            "execution_time_ms": (
                                (run.end_time - run.start_time).total_seconds()
                                * 1000
                                if run.end_time and run.start_time
                                else 0
                            ),
                            "error": run.error,
                            "inputs": run.inputs,
                            "outputs": run.outputs,
                        }
                    )
            except Exception as e:
                logger.warning("meta_agent.trace_fetch_error", error=str(e))

        # Also read local execution logs as fallback
        log_file = Path(os.getenv("FRACTAL_LOGS_DIR", "/app/logs")) / "executions.jsonl"
        if log_file.exists():
            try:
                cutoff = datetime.now(timezone.utc).timestamp() - (
                    time_window_hours * 3600
                )
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        entry_time = datetime.fromisoformat(
                            entry.get("timestamp", "")
                        ).timestamp()
                        if entry_time >= cutoff:
                            traces.append(entry)
            except Exception as e:
                logger.warning("meta_agent.log_read_error", error=str(e))

        return traces

    def _compute_metrics(
        self, traces: list[dict[str, Any]]
    ) -> PerformanceMetrics:
        """Compute performance metrics from traces."""
        if not traces:
            return PerformanceMetrics()

        total = len(traces)
        errors = sum(
            1
            for t in traces
            if t.get("status") in ("failure", "error") or t.get("error")
        )
        times = [
            t.get("execution_time_ms", 0)
            for t in traces
            if t.get("execution_time_ms", 0) > 0
        ]

        avg_time = sum(times) / len(times) if times else 0
        p95_time = sorted(times)[int(len(times) * 0.95)] if times else 0

        # Tool efficiency: ratio of successful completions to total
        successes = total - errors
        tool_efficiency = successes / total if total > 0 else 0

        return PerformanceMetrics(
            error_rate=errors / total if total > 0 else 0,
            avg_completion_time_ms=avg_time,
            p95_completion_time_ms=p95_time,
            tool_efficiency=tool_efficiency,
            total_executions=total,
        )

    def _build_analysis_prompt(
        self,
        metrics: PerformanceMetrics,
        traces: list[dict[str, Any]],
        config: MetaAnalysisInput,
    ) -> str:
        """Build the analysis prompt for the Meta Agent LLM."""
        # Summarize traces (limit to avoid context overflow)
        trace_summaries = []
        for t in traces[:50]:
            trace_summaries.append(
                f"- ID: {t.get('trace_id', 'N/A')}, "
                f"Status: {t.get('status', 'unknown')}, "
                f"Time: {t.get('execution_time_ms', 0):.0f}ms, "
                f"Error: {t.get('error', 'none')}"
            )

        return f"""## Meta-Analysis Cycle {config.analysis_id}

### Current Performance Metrics
- Error Rate: {metrics.error_rate:.2%}
- Avg Completion Time: {metrics.avg_completion_time_ms:.0f}ms
- P95 Completion Time: {metrics.p95_completion_time_ms:.0f}ms
- Tool Efficiency: {metrics.tool_efficiency:.2%}
- Total Executions: {metrics.total_executions}

### Recent Execution Traces ({len(trace_summaries)} of {len(traces)})
{chr(10).join(trace_summaries) if trace_summaries else "No traces available."}

### Tasks
1. Analyze the above metrics and traces for patterns
2. Identify recurring failure modes and successful strategies
3. Propose up to {config.max_proposals} optimization changes
4. Store any valuable strategies in cross-domain memory
5. Archive current config if performance metrics show improvement

Use the available tools to store strategies and archive configurations.
Output your analysis as structured JSON.
"""

    def _parse_agent_output(
        self, output: str
    ) -> tuple[list[FailurePattern], list[OptimizationProposal]]:
        """Parse the agent's text output into structured data."""
        patterns: list[FailurePattern] = []
        proposals: list[OptimizationProposal] = []

        try:
            # Try to parse as JSON
            data = json.loads(output)

            for p in data.get("patterns", []):
                patterns.append(
                    FailurePattern(
                        pattern_type=p.get("type", "unknown"),
                        description=p.get("description", ""),
                        evidence=p.get("evidence", []),
                        frequency=p.get("frequency", 0.0),
                        confidence=p.get("confidence", 0.5),
                    )
                )

            for prop in data.get("proposals", []):
                proposals.append(
                    OptimizationProposal(
                        action=prop.get("action", "modify_config"),
                        target=prop.get("target", ""),
                        change_description=prop.get("change", ""),
                        expected_impact=prop.get("expected_impact", ""),
                        rollback_procedure=prop.get("rollback", "Revert to previous config"),
                        requires_hitl=prop.get("requires_hitl", True),
                        confidence=prop.get("confidence", 0.5),
                    )
                )
        except json.JSONDecodeError:
            # If not valid JSON, create a single pattern from the text
            if output.strip():
                patterns.append(
                    FailurePattern(
                        pattern_type="analysis",
                        description=output[:2000],
                        confidence=0.5,
                    )
                )

        return patterns, proposals

    async def _store_patterns_in_memory(
        self, patterns: list[FailurePattern]
    ) -> int:
        """Store identified patterns in vector memory for future reference."""
        stored = 0
        from src.agents.tools.memory_tools import store_memory

        for pattern in patterns:
            if pattern.confidence >= 0.6:
                try:
                    await store_memory.ainvoke(
                        {
                            "content": (
                                f"[{pattern.pattern_type.upper()}] "
                                f"{pattern.description}"
                            ),
                            "domain": "meta_analysis",
                            "tags": [pattern.pattern_type, "meta_agent"],
                            "importance": min(1.0, pattern.confidence),
                        }
                    )
                    stored += 1
                except Exception as e:
                    logger.warning(
                        "meta_agent.memory_store_error", error=str(e)
                    )

        return stored

    async def _maybe_archive_config(
        self, metrics: PerformanceMetrics
    ) -> int:
        """Archive config if metrics show improvement over baseline."""
        from src.agents.tools.eval_tools import archive_configuration

        # Only archive if error rate is below 10% and we have enough data
        if metrics.error_rate < 0.10 and metrics.total_executions >= 10:
            try:
                await archive_configuration.ainvoke(
                    {
                        "config_name": "auto_archive",
                        "config_data": {
                            "model": os.getenv("VLLM_MODEL_PATH", "default"),
                            "temperature": 0.1,
                            "max_iterations": 25,
                        },
                        "performance_metrics": {
                            "error_rate": metrics.error_rate,
                            "avg_completion_time_ms": metrics.avg_completion_time_ms,
                            "tool_efficiency": metrics.tool_efficiency,
                        },
                        "tags": ["auto", "meta_agent"],
                    }
                )
                return 1
            except Exception as e:
                logger.warning("meta_agent.archive_error", error=str(e))

        return 0


# ── CLI Entry Point ──
if __name__ == "__main__":
    import asyncio

    async def main():
        meta = MetaAgent()
        result = await meta.run_analysis_cycle(
            MetaAnalysisInput(time_window_hours=24)
        )
        print(json.dumps(result.model_dump(), indent=2, default=str))

    asyncio.run(main())
