"""
Fractal — Browser Agent
Async Python client for Browserbase headless browser automation
via Trigger.dev background jobs.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

TRIGGER_API_URL = os.getenv("TRIGGER_API_URL", "https://api.trigger.dev")
TRIGGER_API_KEY = os.getenv("TRIGGER_API_KEY", "")


class BrowserSession(BaseModel):
    """Represents an active Browserbase browser session."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "created"
    url: str | None = None
    page_title: str = ""


class BrowserTaskPayload(BaseModel):
    """Payload for a browser task sent to Trigger.dev."""

    task_type: str
    url: str
    params: dict[str, Any] = Field(default_factory=dict)
    timeout_ms: int = 30000


class BrowserTaskResult(BaseModel):
    """Result from a browser task execution."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str
    status: str = "completed"
    data: dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float = 0.0
    error: str | None = None


class BrowserAgent:
    """
    Manages headless Browserbase sessions through Trigger.dev jobs.

    Provides a high-level interface for:
    - Page navigation and content extraction
    - DOM element interaction (click, type, select)
    - Screenshot capture
    - Multi-page workflows
    """

    def __init__(
        self,
        trigger_api_url: str | None = None,
        trigger_api_key: str | None = None,
    ):
        self.api_url = trigger_api_url or TRIGGER_API_URL
        self.api_key = trigger_api_key or TRIGGER_API_KEY
        self._active_sessions: dict[str, BrowserSession] = {}
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=120.0,
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

    async def _trigger_job(
        self, job_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Trigger a Trigger.dev job and return the result."""
        client = await self._get_client()

        try:
            response = await client.post(
                f"{self.api_url}/api/v1/tasks/trigger",
                json={"id": job_id, "payload": payload},
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(
                "browser_agent.trigger_error",
                job_id=job_id,
                status_code=e.response.status_code,
                detail=e.response.text,
            )
            raise
        except Exception as e:
            logger.error(
                "browser_agent.trigger_error",
                job_id=job_id,
                error=str(e),
            )
            raise

    async def navigate(
        self, url: str, wait_for: str | None = None
    ) -> BrowserTaskResult:
        """
        Navigate to a URL and return page content.
        """
        logger.info("browser_agent.navigate", url=url)

        result = await self._trigger_job(
            "browser-navigate",
            {
                "url": url,
                "waitFor": wait_for,
                "extractText": True,
                "extractLinks": True,
                "extractMetadata": True,
            },
        )

        return BrowserTaskResult(
            task_type="navigate",
            status="completed",
            data=result,
        )

    async def extract_dom(
        self,
        url: str,
        selectors: list[str],
        wait_for: str | None = None,
        timeout_ms: int = 30000,
    ) -> BrowserTaskResult:
        """
        Extract DOM elements from a page by CSS selectors.
        """
        logger.info("browser_agent.extract_dom", url=url, selectors=selectors)

        result = await self._trigger_job(
            "browser-extract-dom",
            {
                "url": url,
                "selectors": selectors,
                "waitFor": wait_for,
                "timeoutMs": timeout_ms,
            },
        )

        return BrowserTaskResult(
            task_type="extract_dom",
            status="completed",
            data=result,
        )

    async def screenshot(
        self,
        url: str,
        full_page: bool = False,
        viewport_width: int = 1280,
        viewport_height: int = 720,
    ) -> BrowserTaskResult:
        """
        Capture a screenshot of a page.
        """
        logger.info("browser_agent.screenshot", url=url, full_page=full_page)

        result = await self._trigger_job(
            "browser-screenshot",
            {
                "url": url,
                "fullPage": full_page,
                "viewportWidth": viewport_width,
                "viewportHeight": viewport_height,
            },
        )

        return BrowserTaskResult(
            task_type="screenshot",
            status="completed",
            data=result,
        )

    async def interact(
        self,
        url: str,
        actions: list[dict[str, Any]],
        timeout_ms: int = 30000,
    ) -> BrowserTaskResult:
        """
        Execute a sequence of interactions on a page.

        Actions format:
        [
            {"type": "click", "selector": "#button"},
            {"type": "type", "selector": "#input", "text": "hello"},
            {"type": "wait", "selector": ".result"},
            {"type": "extract", "selector": ".output"}
        ]
        """
        logger.info(
            "browser_agent.interact",
            url=url,
            action_count=len(actions),
        )

        result = await self._trigger_job(
            "browser-interact",
            {
                "url": url,
                "actions": actions,
                "timeoutMs": timeout_ms,
            },
        )

        return BrowserTaskResult(
            task_type="interact",
            status="completed",
            data=result,
        )
