"""
Fractal — Web Tools
LangChain tool definitions for Browserbase DOM extraction,
page navigation, and screenshot capture.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import structlog
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

TRIGGER_API_URL = os.getenv("TRIGGER_API_URL", "https://api.trigger.dev")
TRIGGER_API_KEY = os.getenv("TRIGGER_API_KEY", "")


# ── Pydantic Input Models ──
class DOMExtractionInput(BaseModel):
    """Input schema for DOM element extraction."""

    url: str = Field(..., description="Target URL to navigate to")
    selectors: list[str] = Field(
        ...,
        description="CSS selectors for elements to extract",
        min_length=1,
        max_length=20,
    )
    wait_for: str | None = Field(
        None, description="CSS selector to wait for before extraction"
    )
    timeout_ms: int = Field(
        default=30000, description="Navigation timeout in milliseconds", ge=1000, le=120000
    )


class ScreenshotInput(BaseModel):
    """Input schema for page screenshot capture."""

    url: str = Field(..., description="Target URL to screenshot")
    full_page: bool = Field(default=False, description="Capture full scrollable page")
    viewport_width: int = Field(default=1280, ge=320, le=3840)
    viewport_height: int = Field(default=720, ge=240, le=2160)


class NavigationInput(BaseModel):
    """Input schema for page navigation and content retrieval."""

    url: str = Field(..., description="Target URL to navigate to")
    extract_text: bool = Field(
        default=True, description="Extract visible text content"
    )
    extract_links: bool = Field(default=False, description="Extract all page links")
    extract_metadata: bool = Field(
        default=False, description="Extract page meta tags"
    )


class DOMExtractionResult(BaseModel):
    """Structured result from DOM extraction."""

    url: str
    elements: list[dict[str, Any]] = Field(default_factory=list)
    extraction_time_ms: float = 0.0
    error: str | None = None


class ScreenshotResult(BaseModel):
    """Structured result from screenshot capture."""

    url: str
    image_base64: str | None = None
    width: int = 0
    height: int = 0
    error: str | None = None


class NavigationResult(BaseModel):
    """Structured result from page navigation."""

    url: str
    title: str = ""
    text_content: str = ""
    links: list[dict[str, str]] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


# ── Tool Functions ──
@tool(args_schema=DOMExtractionInput)
async def extract_dom_elements(
    url: str,
    selectors: list[str],
    wait_for: str | None = None,
    timeout_ms: int = 30000,
) -> dict[str, Any]:
    """
    Extract DOM elements from a web page using headless Browserbase.
    Triggers a background job via Trigger.dev to spin up a browser session,
    navigate to the URL, and extract elements matching the CSS selectors.
    Returns structured data for each matched element.
    """
    logger.info("web_tools.extract_dom", url=url, selectors=selectors)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{TRIGGER_API_URL}/api/v1/tasks/trigger",
                headers={
                    "Authorization": f"Bearer {TRIGGER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "id": "browser-extract-dom",
                    "payload": {
                        "url": url,
                        "selectors": selectors,
                        "waitFor": wait_for,
                        "timeoutMs": timeout_ms,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

        result = DOMExtractionResult(
            url=url,
            elements=data.get("elements", []),
            extraction_time_ms=data.get("extractionTimeMs", 0),
        )
        return result.model_dump()

    except Exception as e:
        logger.error("web_tools.extract_dom.error", url=url, error=str(e))
        result = DOMExtractionResult(url=url, error=str(e))
        return result.model_dump()


@tool(args_schema=ScreenshotInput)
async def capture_screenshot(
    url: str,
    full_page: bool = False,
    viewport_width: int = 1280,
    viewport_height: int = 720,
) -> dict[str, Any]:
    """
    Capture a screenshot of a web page using headless Browserbase.
    Returns the screenshot as a base64-encoded PNG image.
    """
    logger.info("web_tools.screenshot", url=url, full_page=full_page)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{TRIGGER_API_URL}/api/v1/tasks/trigger",
                headers={
                    "Authorization": f"Bearer {TRIGGER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "id": "browser-screenshot",
                    "payload": {
                        "url": url,
                        "fullPage": full_page,
                        "viewportWidth": viewport_width,
                        "viewportHeight": viewport_height,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

        result = ScreenshotResult(
            url=url,
            image_base64=data.get("imageBase64"),
            width=data.get("width", viewport_width),
            height=data.get("height", viewport_height),
        )
        return result.model_dump()

    except Exception as e:
        logger.error("web_tools.screenshot.error", url=url, error=str(e))
        result = ScreenshotResult(url=url, error=str(e))
        return result.model_dump()


@tool(args_schema=NavigationInput)
async def navigate_and_extract(
    url: str,
    extract_text: bool = True,
    extract_links: bool = False,
    extract_metadata: bool = False,
) -> dict[str, Any]:
    """
    Navigate to a URL and extract page content.
    Can extract visible text, all links, and meta tags.
    Uses Browserbase for full JavaScript rendering.
    """
    logger.info("web_tools.navigate", url=url)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{TRIGGER_API_URL}/api/v1/tasks/trigger",
                headers={
                    "Authorization": f"Bearer {TRIGGER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "id": "browser-navigate",
                    "payload": {
                        "url": url,
                        "extractText": extract_text,
                        "extractLinks": extract_links,
                        "extractMetadata": extract_metadata,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

        result = NavigationResult(
            url=url,
            title=data.get("title", ""),
            text_content=data.get("textContent", ""),
            links=data.get("links", []),
            metadata=data.get("metadata", {}),
        )
        return result.model_dump()

    except Exception as e:
        logger.error("web_tools.navigate.error", url=url, error=str(e))
        result = NavigationResult(url=url, error=str(e))
        return result.model_dump()


def get_web_tools() -> list:
    """Return all web-related LangChain tools."""
    return [extract_dom_elements, capture_screenshot, navigate_and_extract]
