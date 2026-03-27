"""
Fractal — Task Agent
Primary LangChain orchestrator with behavioral guardrails,
structured tool usage, and HITL integration.
"""

from __future__ import annotations

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
from langsmith import traceable
from pydantic import BaseModel, Field

from src.agents.tools.eval_tools import get_eval_tools
from src.agents.tools.memory_tools import get_memory_tools
from src.agents.tools.web_tools import get_web_tools
from src.middleware.hitl_gate import HITLGate

load_dotenv()
logger = structlog.get_logger(__name__)

# ── Constants ──
PROMPTS_DIR = Path(__file__).parent / "prompts"
VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_PORT = os.getenv("VLLM_PORT", "8000")
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"


# ── Pydantic Models ──
class TaskInput(BaseModel):
    """Validated input for a task execution request."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    objective: str = Field(..., min_length=1, max_length=10000)
    context: dict[str, Any] = Field(default_factory=dict)
    max_steps: int = Field(default=25, ge=1, le=100)
    require_hitl: bool = Field(default=False)
    priority: str = Field(default="normal", pattern=r"^(low|normal|high|critical)$")


class TaskResult(BaseModel):
    """Structured output from a completed task."""

    task_id: str
    status: str = Field(pattern=r"^(success|failure|pending_hitl|timeout)$")
    result: dict[str, Any] = Field(default_factory=dict)
    steps_taken: int = 0
    tools_used: list[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    errors: list[str] = Field(default_factory=list)


class TaskAgent:
    """
    Primary Fractal orchestrator powered by LangChain.

    Connects to the local vLLM inference engine (Intel Arc Pro B70),
    enforces behavioral guardrails via the system prompt, and
    integrates HITL approval gates for destructive actions.
    """

    def __init__(
        self,
        model_name: str = "default",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.agent_id = str(uuid.uuid4())
        self.hitl_gate = HITLGate()

        # Load system prompt
        system_prompt_path = PROMPTS_DIR / "task_system.txt"
        self.system_prompt = system_prompt_path.read_text(encoding="utf-8")

        # Initialize LLM (pointed at local vLLM)
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key="EMPTY",
            openai_api_base=VLLM_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs={"top_p": 0.95},
        )

        # Collect all tools
        self.tools = [
            *get_web_tools(),
            *get_memory_tools(),
            *get_eval_tools(),
        ]

        # Create the react agent graph using langgraph
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=self.system_prompt,
        )

        logger.info(
            "task_agent.initialized",
            agent_id=self.agent_id,
            model=model_name,
            vllm_url=VLLM_BASE_URL,
            tool_count=len(self.tools),
        )

    @traceable(name="task_agent.execute", run_type="chain")
    async def execute(self, task: TaskInput) -> TaskResult:
        """
        Execute a task with full tracing, guardrails, and HITL checks.
        """
        start_time = datetime.now(timezone.utc)
        tools_used: list[str] = []
        errors: list[str] = []

        logger.info(
            "task_agent.execute.start",
            task_id=task.task_id,
            objective=task.objective[:200],
            priority=task.priority,
        )

        try:
            # Check if HITL pre-approval is required
            if task.require_hitl:
                approval = await self.hitl_gate.request_approval(
                    action_type="task_execution",
                    description=f"Execute task: {task.objective[:200]}",
                    metadata={"task_id": task.task_id, "priority": task.priority},
                )
                if not approval.approved:
                    return TaskResult(
                        task_id=task.task_id,
                        status="pending_hitl",
                        result={"reason": approval.reason},
                        confidence=0.0,
                    )

            # Build input with context
            input_text = self._format_input(task)

            # Execute the agent
            result = await self.agent.ainvoke(
                {"messages": [{"role": "user", "content": input_text}]},
            )

            # Extract tools used from messages
            messages = result.get("messages", [])
            for msg in messages:
                if hasattr(msg, "tool_calls"):
                    for tc in getattr(msg, "tool_calls", []):
                        tools_used.append(tc.get("name", ""))

            # Calculate execution time
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            return TaskResult(
                task_id=task.task_id,
                status="success",
                result={"output": str(messages[-1].content) if messages else ""},
                steps_taken=len(messages),
                tools_used=list(set(tools_used)),
                execution_time_ms=elapsed,
                confidence=0.85,
            )

        except Exception as e:
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            error_msg = f"{type(e).__name__}: {str(e)}"
            errors.append(error_msg)

            logger.error(
                "task_agent.execute.error",
                task_id=task.task_id,
                error=error_msg,
            )

            return TaskResult(
                task_id=task.task_id,
                status="failure",
                result={"error": error_msg},
                steps_taken=0,
                tools_used=tools_used,
                execution_time_ms=elapsed,
                confidence=0.0,
                errors=errors,
            )

    def _format_input(self, task: TaskInput) -> str:
        """Format task input with context for the agent."""
        parts = [f"## Task Objective\n{task.objective}"]

        if task.context:
            context_str = "\n".join(
                f"- **{k}**: {v}" for k, v in task.context.items()
            )
            parts.append(f"\n## Context\n{context_str}")

        parts.append(f"\n## Constraints\n- Maximum steps: {task.max_steps}")
        parts.append(f"- Priority: {task.priority}")

        return "\n".join(parts)

    @traceable(name="task_agent.health_check")
    async def health_check(self) -> dict[str, Any]:
        """Check agent and inference engine health."""
        try:
            # Quick inference test
            test_result = await self.llm.ainvoke("Respond with 'OK'.")
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "inference": "connected",
                "tool_count": len(self.tools),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "agent_id": self.agent_id,
                "inference": f"error: {e}",
                "tool_count": len(self.tools),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


# ── CLI Entry Point ──
if __name__ == "__main__":
    import asyncio

    async def main():
        agent = TaskAgent()
        health = await agent.health_check()
        logger.info("task_agent.health", **health)

        if health["status"] == "healthy":
            logger.info("task_agent.ready", agent_id=agent.agent_id)
        else:
            logger.error("task_agent.startup_failed", **health)

    asyncio.run(main())
