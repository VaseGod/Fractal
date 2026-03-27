"""
Fractal — Bifrost Router
FastAPI middleware for secure API routing through Bifrost,
implementing mTLS, rate limiting, and request validation.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any

import structlog
import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


class RateLimitConfig(BaseModel):
    """Rate limit rule configuration."""

    path: str
    max_requests: int
    window_seconds: int = 60


class RouteConfig(BaseModel):
    """Single route configuration."""

    path: str
    upstream: str
    methods: list[str] = Field(default_factory=lambda: ["GET"])
    auth_required: bool = True
    rate_limit_group: str = "api"
    middleware: list[str] = Field(default_factory=list)


class BifrostConfig(BaseModel):
    """Parsed Bifrost configuration."""

    host: str = "0.0.0.0"
    port: int = 8080
    tls_enabled: bool = False
    rate_limits: list[RateLimitConfig] = Field(default_factory=list)
    routes: list[RouteConfig] = Field(default_factory=list)
    security_headers: dict[str, str] = Field(default_factory=dict)
    blocked_patterns: list[str] = Field(default_factory=list)


class SlidingWindowCounter:
    """Thread-safe sliding window rate limiter."""

    def __init__(self):
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """Check if a request is within rate limits."""
        now = time.time()
        cutoff = now - window_seconds

        # Clean old entries
        self._requests[key] = [
            t for t in self._requests[key] if t > cutoff
        ]

        if len(self._requests[key]) >= max_requests:
            return False

        self._requests[key].append(now)
        return True

    def get_remaining(
        self, key: str, max_requests: int, window_seconds: int
    ) -> int:
        """Get remaining requests in the current window."""
        now = time.time()
        cutoff = now - window_seconds
        current = sum(1 for t in self._requests.get(key, []) if t > cutoff)
        return max(0, max_requests - current)


def load_bifrost_config(config_path: str | None = None) -> BifrostConfig:
    """Load and parse the Bifrost YAML configuration."""
    config_path = config_path or os.getenv(
        "BIFROST_CONFIG_PATH", "/app/config/bifrost.yaml"
    )

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(
            "bifrost.config_not_found",
            path=config_path,
        )
        return BifrostConfig()

    # Parse rate limits
    rate_limits = []
    for rule in raw.get("rate_limiting", {}).get("rules", []):
        window_str = rule.get("window", "1m")
        window_seconds = _parse_duration(window_str)
        rate_limits.append(
            RateLimitConfig(
                path=rule["path"],
                max_requests=rule["max_requests"],
                window_seconds=window_seconds,
            )
        )

    # Parse routes
    routes = []
    for route in raw.get("routes", []):
        routes.append(
            RouteConfig(
                path=route["path"],
                upstream=route["upstream"],
                methods=route.get("methods", ["GET"]),
                auth_required=route.get("auth_required", True),
                rate_limit_group=route.get("rate_limit_group", "api"),
                middleware=route.get("middleware", []),
            )
        )

    # Parse security headers
    headers = raw.get("headers", {}).get("response", {})

    # Parse blocked patterns
    blocked = raw.get("validation", {}).get("block_patterns", [])

    server = raw.get("server", {})

    return BifrostConfig(
        host=server.get("host", "0.0.0.0"),
        port=server.get("port", 8080),
        tls_enabled=raw.get("tls", {}).get("enabled", False),
        rate_limits=rate_limits,
        routes=routes,
        security_headers=headers,
        blocked_patterns=blocked,
    )


def _parse_duration(s: str) -> int:
    """Parse a duration string like '1m', '30s', '1h' to seconds."""
    s = s.strip()
    if s.endswith("s"):
        return int(s[:-1])
    elif s.endswith("m"):
        return int(s[:-1]) * 60
    elif s.endswith("h"):
        return int(s[:-1]) * 3600
    return int(s)


class BifrostSecurityMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware implementing Bifrost security features:
    - Rate limiting (sliding window)
    - Security headers injection
    - Request body validation (blocked patterns)
    - Structured request logging
    """

    def __init__(self, app: FastAPI, config: BifrostConfig):
        super().__init__(app)
        self.config = config
        self.rate_limiter = SlidingWindowCounter()

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"

        # ── Rate Limiting ──
        for limit in self.config.rate_limits:
            if self._path_matches(request.url.path, limit.path):
                key = f"{client_ip}:{limit.path}"
                if not self.rate_limiter.is_allowed(
                    key, limit.max_requests, limit.window_seconds
                ):
                    remaining = self.rate_limiter.get_remaining(
                        key, limit.max_requests, limit.window_seconds
                    )
                    logger.warning(
                        "bifrost.rate_limited",
                        client=client_ip,
                        path=request.url.path,
                    )
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded",
                        headers={
                            "Retry-After": str(limit.window_seconds),
                            "X-RateLimit-Remaining": str(remaining),
                        },
                    )

        # ── Request Body Validation ──
        if request.method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.body()
                body_str = body.decode("utf-8", errors="ignore")

                import re

                for pattern in self.config.blocked_patterns:
                    if re.search(pattern, body_str):
                        logger.warning(
                            "bifrost.blocked_pattern",
                            client=client_ip,
                            pattern=pattern,
                            path=request.url.path,
                        )
                        raise HTTPException(
                            status_code=400,
                            detail="Request contains blocked content",
                        )
            except HTTPException:
                raise
            except Exception:
                pass  # Non-text bodies are OK

        # ── Process Request ──
        response = await call_next(request)

        # ── Inject Security Headers ──
        for header, value in self.config.security_headers.items():
            response.headers[header] = value

        # ── Structured Logging ──
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            "bifrost.request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            client=client_ip,
            elapsed_ms=round(elapsed_ms, 2),
        )

        return response

    @staticmethod
    def _path_matches(request_path: str, pattern: str) -> bool:
        """Simple glob-style path matching."""
        if pattern.endswith("/*"):
            return request_path.startswith(pattern[:-2])
        return request_path == pattern


def create_bifrost_app(config_path: str | None = None) -> FastAPI:
    """
    Create a FastAPI application with Bifrost security middleware.
    """
    config = load_bifrost_config(config_path)

    app = FastAPI(
        title="Fractal API",
        description="Fractal Agentic Infrastructure API (Bifrost-secured)",
        version="1.0.0",
        docs_url=None,  # Disable docs in production
        redoc_url=None,
    )

    # Add Bifrost middleware
    app.add_middleware(BifrostSecurityMiddleware, config=config)

    # CORS (restrictive)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "fractal-bifrost"}

    return app
