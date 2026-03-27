"""
Fractal — Test Configuration
Shared fixtures and configuration for the test suite.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Set test environment variables."""
    monkeypatch.setenv("VLLM_HOST", "localhost")
    monkeypatch.setenv("VLLM_PORT", "8000")
    monkeypatch.setenv("CHROMA_HOST", "localhost")
    monkeypatch.setenv("CHROMA_PORT", "8100")
    monkeypatch.setenv("CHROMA_COLLECTION", "fractal_test")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls_test_key")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("TRIGGER_API_KEY", "tr_test_key")
    monkeypatch.setenv("BROWSERBASE_API_KEY", "bb_test_key")
    monkeypatch.setenv("FRACTAL_DATA_DIR", "/tmp/fractal-test-data")
    monkeypatch.setenv("FRACTAL_LOGS_DIR", "/tmp/fractal-test-logs")
    monkeypatch.setenv("ARC_AGI_API_KEY", "arc_test_key")


@pytest.fixture
def mock_vector_store():
    """Provide a mock vector store."""
    store = MagicMock()
    store.add = AsyncMock(return_value="test-id-123")
    store.query = AsyncMock(
        return_value={
            "documents": [["test document"]],
            "metadatas": [
                [
                    {
                        "domain": "general",
                        "tags": "test",
                        "importance": "0.5",
                        "created_at": "2026-01-01T00:00:00+00:00",
                    }
                ]
            ],
            "distances": [[0.2]],
            "ids": [["test-id"]],
        }
    )
    store.count = AsyncMock(return_value=10)
    store.delete = AsyncMock()
    store.remove_stale = AsyncMock(return_value=3)
    return store


@pytest.fixture
def mock_httpx_client():
    """Provide a mock httpx async client."""
    client = AsyncMock()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"status": "ok"}
    response.raise_for_status = MagicMock()
    client.post = AsyncMock(return_value=response)
    client.get = AsyncMock(return_value=response)
    return client
