"""
Fractal — Evaluation Tools
LangChain tools for the Meta Agent to analyze execution logs,
modify evaluation scripts, and archive configurations.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

DATA_DIR = Path(os.getenv("FRACTAL_DATA_DIR", "/app/data"))
LOGS_DIR = Path(os.getenv("FRACTAL_LOGS_DIR", "/app/logs"))
ARCHIVE_DIR = DATA_DIR / "config_archive"
EVAL_SCRIPTS_DIR = DATA_DIR / "eval_scripts"


# ── Pydantic Models ──
class ExecutionLogAnalysisInput(BaseModel):
    """Input for analyzing execution logs."""

    trace_ids: list[str] | None = Field(
        None, description="Specific trace IDs to analyze (None = latest batch)"
    )
    time_window_hours: int = Field(
        default=24,
        description="How far back to look for logs (hours)",
        ge=1,
        le=720,
    )
    metrics: list[str] = Field(
        default=["error_rate", "completion_time", "tool_efficiency"],
        description="Which metrics to compute",
    )


class UpdateEvalScriptInput(BaseModel):
    """Input for modifying an evaluation script."""

    script_name: str = Field(
        ..., description="Name of the evaluation script to modify"
    )
    modification: str = Field(
        ..., description="Description of the modification to make"
    )
    new_content: str = Field(
        ..., description="The new script content (Python code)"
    )
    justification: str = Field(
        ..., description="Why this modification improves evaluation"
    )


class ArchiveConfigInput(BaseModel):
    """Input for archiving a successful configuration."""

    config_name: str = Field(
        ..., description="Name for this configuration snapshot"
    )
    config_data: dict[str, Any] = Field(
        ..., description="The configuration data to archive"
    )
    performance_metrics: dict[str, float] = Field(
        ..., description="Performance metrics that justify archiving"
    )
    tags: list[str] = Field(
        default_factory=list, description="Tags for categorization"
    )


class LogAnalysisResult(BaseModel):
    """Result from execution log analysis."""

    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    time_window_hours: int
    total_executions: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    failure_patterns: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class UpdateEvalResult(BaseModel):
    """Result from updating an evaluation script."""

    script_name: str
    version: int = 0
    status: str = "updated"
    backup_path: str = ""
    requires_hitl: bool = True


class ArchiveConfigResult(BaseModel):
    """Result from archiving a configuration."""

    archive_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    config_name: str
    archive_path: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Tool Functions ──
@tool(args_schema=ExecutionLogAnalysisInput)
async def analyze_execution_log(
    trace_ids: list[str] | None = None,
    time_window_hours: int = 24,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """
    Analyze Task Agent execution logs to compute performance metrics.
    Examines error rates, completion times, tool usage efficiency,
    and identifies recurring failure patterns. Results feed into
    the Meta Agent's optimization pipeline.
    """
    metrics = metrics or ["error_rate", "completion_time", "tool_efficiency"]
    logger.info(
        "eval_tools.analyze_log",
        time_window_hours=time_window_hours,
        metrics=metrics,
    )

    try:
        # Read execution logs
        log_entries = _read_execution_logs(time_window_hours)

        # Compute requested metrics
        computed_metrics: dict[str, float] = {}
        failure_patterns: list[dict[str, Any]] = []
        recommendations: list[str] = []

        total = len(log_entries)
        if total == 0:
            return LogAnalysisResult(
                time_window_hours=time_window_hours,
                total_executions=0,
                metrics={},
                recommendations=["No execution logs found in window."],
            ).model_dump()

        if "error_rate" in metrics:
            errors = sum(1 for e in log_entries if e.get("status") == "failure")
            computed_metrics["error_rate"] = errors / total if total > 0 else 0.0

            if computed_metrics["error_rate"] > 0.2:
                recommendations.append(
                    f"Error rate is {computed_metrics['error_rate']:.1%} — "
                    "investigate common failure modes."
                )

        if "completion_time" in metrics:
            times = [
                e.get("execution_time_ms", 0)
                for e in log_entries
                if e.get("execution_time_ms")
            ]
            if times:
                computed_metrics["avg_completion_time_ms"] = sum(times) / len(times)
                computed_metrics["p95_completion_time_ms"] = sorted(times)[
                    int(len(times) * 0.95)
                ]

        if "tool_efficiency" in metrics:
            tool_calls = sum(e.get("steps_taken", 0) for e in log_entries)
            successes = sum(1 for e in log_entries if e.get("status") == "success")
            computed_metrics["tool_efficiency"] = (
                successes / tool_calls if tool_calls > 0 else 0.0
            )

        # Identify failure patterns
        error_types: dict[str, int] = {}
        for entry in log_entries:
            if entry.get("status") == "failure":
                for error in entry.get("errors", []):
                    error_type = error.split(":")[0] if ":" in error else error
                    error_types[error_type] = error_types.get(error_type, 0) + 1

        for error_type, count in sorted(
            error_types.items(), key=lambda x: x[1], reverse=True
        ):
            failure_patterns.append(
                {
                    "type": error_type,
                    "count": count,
                    "frequency": count / total,
                }
            )

        result = LogAnalysisResult(
            time_window_hours=time_window_hours,
            total_executions=total,
            metrics=computed_metrics,
            failure_patterns=failure_patterns[:10],
            recommendations=recommendations,
        )
        return result.model_dump()

    except Exception as e:
        logger.error("eval_tools.analyze_log.error", error=str(e))
        return LogAnalysisResult(
            time_window_hours=time_window_hours,
            recommendations=[f"Analysis failed: {e}"],
        ).model_dump()


@tool(args_schema=UpdateEvalScriptInput)
async def update_evaluation_script(
    script_name: str,
    modification: str,
    new_content: str,
    justification: str,
) -> dict[str, Any]:
    """
    Modify an evaluation script used by the Meta Agent.
    Creates a versioned backup before modification.
    REQUIRES HITL APPROVAL before changes take effect.
    The Meta Agent uses this to evolve its own evaluation criteria.
    """
    logger.info(
        "eval_tools.update_script",
        script_name=script_name,
        modification=modification[:100],
    )

    try:
        EVAL_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        script_path = EVAL_SCRIPTS_DIR / f"{script_name}.py"

        # Version tracking
        version = 1
        if script_path.exists():
            # Create versioned backup
            backup_dir = EVAL_SCRIPTS_DIR / "backups"
            backup_dir.mkdir(exist_ok=True)
            version = len(list(backup_dir.glob(f"{script_name}_v*.py"))) + 1
            backup_path = backup_dir / f"{script_name}_v{version}.py"
            shutil.copy2(script_path, backup_path)
        else:
            backup_path = Path("")

        # Write new content (pending HITL approval)
        pending_path = EVAL_SCRIPTS_DIR / f"{script_name}.pending.py"
        pending_path.write_text(new_content, encoding="utf-8")

        # Write modification metadata
        meta_path = EVAL_SCRIPTS_DIR / f"{script_name}.pending.json"
        meta_path.write_text(
            json.dumps(
                {
                    "script_name": script_name,
                    "version": version + 1,
                    "modification": modification,
                    "justification": justification,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "pending_hitl",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        result = UpdateEvalResult(
            script_name=script_name,
            version=version + 1,
            status="pending_hitl",
            backup_path=str(backup_path),
            requires_hitl=True,
        )
        return result.model_dump()

    except Exception as e:
        logger.error("eval_tools.update_script.error", error=str(e))
        return {
            "script_name": script_name,
            "status": f"error: {e}",
            "requires_hitl": True,
        }


@tool(args_schema=ArchiveConfigInput)
async def archive_configuration(
    config_name: str,
    config_data: dict[str, Any],
    performance_metrics: dict[str, float],
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Archive a successful agent configuration for future iterations.
    Stores the configuration along with its performance metrics,
    enabling the Meta Agent to retrieve and compare past configurations
    when optimizing the system.
    """
    tags = tags or []
    logger.info("eval_tools.archive_config", config_name=config_name)

    try:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

        archive_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        archive_entry = {
            "archive_id": archive_id,
            "config_name": config_name,
            "config_data": config_data,
            "performance_metrics": performance_metrics,
            "tags": tags,
            "timestamp": timestamp,
        }

        archive_path = ARCHIVE_DIR / f"{config_name}_{archive_id[:8]}.json"
        archive_path.write_text(
            json.dumps(archive_entry, indent=2, default=str),
            encoding="utf-8",
        )

        result = ArchiveConfigResult(
            archive_id=archive_id,
            config_name=config_name,
            archive_path=str(archive_path),
        )
        return result.model_dump()

    except Exception as e:
        logger.error("eval_tools.archive_config.error", error=str(e))
        return {"config_name": config_name, "status": f"error: {e}"}


# ── Internal Helpers ──
def _read_execution_logs(time_window_hours: int) -> list[dict[str, Any]]:
    """Read execution log entries from the log directory."""
    entries: list[dict[str, Any]] = []

    log_file = LOGS_DIR / "executions.jsonl"
    if not log_file.exists():
        return entries

    cutoff = datetime.now(timezone.utc).timestamp() - (time_window_hours * 3600)

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entry_time = datetime.fromisoformat(
                        entry.get("timestamp", "")
                    ).timestamp()
                    if entry_time >= cutoff:
                        entries.append(entry)
                except (json.JSONDecodeError, ValueError):
                    continue
    except Exception as e:
        logger.warning("eval_tools.read_logs.error", error=str(e))

    return entries


def get_eval_tools() -> list:
    """Return all evaluation-related LangChain tools."""
    return [analyze_execution_log, update_evaluation_script, archive_configuration]
