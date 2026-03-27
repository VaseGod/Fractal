"""
Fractal — HITL (Human-in-the-Loop) Gate
LangSmith-integrated approval system that pauses execution
for explicit human approval before destructive actions.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog
from langsmith import Client as LangSmithClient
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ActionType(str, Enum):
    """Types of actions that can trigger HITL approval."""

    DATABASE_WRITE = "database_write"
    DATABASE_DELETE = "database_delete"
    EXTERNAL_API_CALL = "external_api_call"
    FILE_SYSTEM_WRITE = "file_system_write"
    CONFIGURATION_MODIFY = "configuration_modify"
    EVALUATION_SCRIPT_MODIFY = "evaluation_script_modify"
    MEMORY_PURGE = "memory_purge"
    TASK_EXECUTION = "task_execution"


class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    ERROR = "error"


class ApprovalRequest(BaseModel):
    """A request for human approval."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action_type: str
    description: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    timeout_seconds: int = 300


class ApprovalResponse(BaseModel):
    """Response from the HITL approval system."""

    request_id: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved: bool = False
    reason: str = ""
    reviewer: str = ""
    reviewed_at: str | None = None


# Safe actions that bypass HITL
SAFE_ACTIONS = frozenset(
    {
        "memory_read",
        "memory_query",
        "log_analysis",
        "score_calculation",
        "dom_extraction",
        "screenshot_capture",
        "page_navigation",
    }
)

# Actions that ALWAYS require approval
DESTRUCTIVE_ACTIONS = frozenset(
    {
        ActionType.DATABASE_WRITE,
        ActionType.DATABASE_DELETE,
        ActionType.EXTERNAL_API_CALL,
        ActionType.FILE_SYSTEM_WRITE,
        ActionType.CONFIGURATION_MODIFY,
        ActionType.EVALUATION_SCRIPT_MODIFY,
        ActionType.MEMORY_PURGE,
    }
)


class HITLGate:
    """
    Human-in-the-Loop approval gate.

    Integrates with LangSmith to create approval requests as
    annotations on runs. Blocks execution until a human reviewer
    approves or rejects the action, or until timeout.
    """

    def __init__(self):
        self.timeout_seconds = int(
            os.getenv("HITL_TIMEOUT_SECONDS", "300")
        )
        self.langsmith_client: LangSmithClient | None = None
        self._pending_requests: dict[str, ApprovalRequest] = {}

        # Initialize LangSmith client if configured
        api_key = os.getenv("LANGCHAIN_API_KEY")
        if api_key and api_key != "ls_your_api_key_here":
            try:
                self.langsmith_client = LangSmithClient(api_key=api_key)
                logger.info("hitl_gate.langsmith_connected")
            except Exception as e:
                logger.warning(
                    "hitl_gate.langsmith_connection_failed", error=str(e)
                )

    def is_safe_action(self, action_type: str) -> bool:
        """Check if an action type is considered safe (no HITL needed)."""
        return action_type in SAFE_ACTIONS

    def requires_approval(self, action_type: str) -> bool:
        """Check if an action type requires HITL approval."""
        if self.is_safe_action(action_type):
            return False
        return True

    async def request_approval(
        self,
        action_type: str,
        description: str,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> ApprovalResponse:
        """
        Request human approval for an action.

        If the action is safe, auto-approves immediately.
        Otherwise, creates a LangSmith annotation and waits
        for human review up to the configured timeout.
        """
        metadata = metadata or {}

        # Auto-approve safe actions
        if self.is_safe_action(action_type):
            logger.debug("hitl_gate.auto_approved", action_type=action_type)
            return ApprovalResponse(
                request_id=str(uuid.uuid4()),
                status=ApprovalStatus.APPROVED,
                approved=True,
                reason="Safe action — auto-approved",
                reviewer="system",
                reviewed_at=datetime.now(timezone.utc).isoformat(),
            )

        # Create approval request
        request = ApprovalRequest(
            action_type=action_type,
            description=description,
            metadata=metadata,
            timeout_seconds=self.timeout_seconds,
        )

        logger.info(
            "hitl_gate.approval_requested",
            request_id=request.request_id,
            action_type=action_type,
            description=description[:200],
        )

        self._pending_requests[request.request_id] = request

        # Create LangSmith annotation if available
        if self.langsmith_client and run_id:
            try:
                self.langsmith_client.create_feedback(
                    run_id=run_id,
                    key="hitl_approval_request",
                    score=0,  # Pending
                    comment=(
                        f"HITL Approval Required\n"
                        f"Action: {action_type}\n"
                        f"Description: {description}\n"
                        f"Request ID: {request.request_id}"
                    ),
                )
            except Exception as e:
                logger.warning(
                    "hitl_gate.langsmith_annotation_failed", error=str(e)
                )

        # In production, this would poll for approval.
        # For now, we return a pending status that the orchestrator handles.
        return ApprovalResponse(
            request_id=request.request_id,
            status=ApprovalStatus.PENDING,
            approved=False,
            reason=f"Awaiting human approval (timeout: {self.timeout_seconds}s)",
        )

    async def check_approval(self, request_id: str) -> ApprovalResponse:
        """Check the current status of an approval request."""
        if request_id not in self._pending_requests:
            return ApprovalResponse(
                request_id=request_id,
                status=ApprovalStatus.ERROR,
                approved=False,
                reason="Unknown request ID",
            )

        request = self._pending_requests[request_id]

        # Check for timeout
        request_time = datetime.fromisoformat(request.timestamp)
        elapsed = (datetime.now(timezone.utc) - request_time).total_seconds()

        if elapsed > request.timeout_seconds:
            del self._pending_requests[request_id]
            return ApprovalResponse(
                request_id=request_id,
                status=ApprovalStatus.TIMEOUT,
                approved=False,
                reason=f"Approval timed out after {request.timeout_seconds}s",
            )

        # Check LangSmith for feedback
        if self.langsmith_client:
            try:
                # In production, query LangSmith for feedback on this request
                pass
            except Exception as e:
                logger.warning(
                    "hitl_gate.check_approval_error", error=str(e)
                )

        return ApprovalResponse(
            request_id=request_id,
            status=ApprovalStatus.PENDING,
            approved=False,
            reason="Still awaiting approval",
        )

    async def approve(
        self, request_id: str, reviewer: str = "admin", reason: str = ""
    ) -> ApprovalResponse:
        """Manually approve a pending request (called by review API)."""
        if request_id in self._pending_requests:
            del self._pending_requests[request_id]

        return ApprovalResponse(
            request_id=request_id,
            status=ApprovalStatus.APPROVED,
            approved=True,
            reason=reason or "Approved by reviewer",
            reviewer=reviewer,
            reviewed_at=datetime.now(timezone.utc).isoformat(),
        )

    async def reject(
        self, request_id: str, reviewer: str = "admin", reason: str = ""
    ) -> ApprovalResponse:
        """Manually reject a pending request (called by review API)."""
        if request_id in self._pending_requests:
            del self._pending_requests[request_id]

        return ApprovalResponse(
            request_id=request_id,
            status=ApprovalStatus.REJECTED,
            approved=False,
            reason=reason or "Rejected by reviewer",
            reviewer=reviewer,
            reviewed_at=datetime.now(timezone.utc).isoformat(),
        )
