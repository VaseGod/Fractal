"""
Fractal — Memory Tools
LangChain tools for vector database read/write operations
using ChromaDB and the MemCollab architecture.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import structlog
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# Lazy imports — these are initialized when tools are first called
_vector_store = None


def _get_vector_store():
    """Lazy-initialize the vector store singleton."""
    global _vector_store
    if _vector_store is None:
        from src.memory.vector_store import VectorStore

        _vector_store = VectorStore()
    return _vector_store


# ── Pydantic Input/Output Models ──
class StoreMemoryInput(BaseModel):
    """Input schema for storing a memory entry."""

    content: str = Field(
        ..., description="The content to store in memory", min_length=1, max_length=50000
    )
    domain: str = Field(
        default="general",
        description="Knowledge domain (e.g., 'web_traversal', 'evaluation', 'strategy')",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization",
        max_length=10,
    )
    importance: float = Field(
        default=0.5,
        description="Importance score (0.0 = trivial, 1.0 = critical)",
        ge=0.0,
        le=1.0,
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata key-value pairs",
    )


class QueryMemoryInput(BaseModel):
    """Input schema for querying memory."""

    query: str = Field(
        ..., description="Natural language query to search memory", min_length=1
    )
    top_k: int = Field(
        default=5, description="Number of results to return", ge=1, le=50
    )
    domain: str | None = Field(
        None, description="Filter by knowledge domain"
    )
    min_importance: float = Field(
        default=0.0,
        description="Minimum importance score filter",
        ge=0.0,
        le=1.0,
    )


class ForgetStaleInput(BaseModel):
    """Input schema for forgetting stale memory entries."""

    max_age_days: int = Field(
        default=30,
        description="Maximum age in days before a memory is considered stale",
        ge=1,
        le=365,
    )
    domain: str | None = Field(
        None, description="Only forget stale entries in this domain"
    )
    min_importance_to_keep: float = Field(
        default=0.8,
        description="Entries with importance >= this value are never forgotten",
        ge=0.0,
        le=1.0,
    )


class MemoryEntry(BaseModel):
    """A single memory entry returned from a query."""

    id: str
    content: str
    domain: str
    tags: list[str]
    importance: float
    similarity_score: float = 0.0
    created_at: str
    metadata: dict[str, str] = Field(default_factory=dict)


class QueryMemoryResult(BaseModel):
    """Structured result from a memory query."""

    query: str
    results: list[MemoryEntry] = Field(default_factory=list)
    total_found: int = 0
    search_time_ms: float = 0.0


class StoreMemoryResult(BaseModel):
    """Result from storing a memory entry."""

    memory_id: str
    status: str = "stored"
    domain: str = ""


class ForgetStaleResult(BaseModel):
    """Result from forgetting stale memories."""

    removed_count: int = 0
    remaining_count: int = 0
    domain: str | None = None


# ── Tool Functions ──
@tool(args_schema=StoreMemoryInput)
async def store_memory(
    content: str,
    domain: str = "general",
    tags: list[str] | None = None,
    importance: float = 0.5,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Store a piece of knowledge in the cross-domain vector memory.
    Entries are embedded and indexed for semantic retrieval.
    Use domains to organize knowledge (e.g., 'web_traversal', 'strategy').
    High-importance entries (>= 0.8) are protected from automatic cleanup.
    """
    tags = tags or []
    metadata = metadata or {}

    logger.info("memory_tools.store", domain=domain, importance=importance)

    try:
        store = _get_vector_store()

        full_metadata = {
            **metadata,
            "domain": domain,
            "tags": ",".join(tags),
            "importance": str(importance),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        memory_id = await store.add(
            content=content,
            metadata=full_metadata,
        )

        result = StoreMemoryResult(
            memory_id=memory_id, status="stored", domain=domain
        )
        return result.model_dump()

    except Exception as e:
        logger.error("memory_tools.store.error", error=str(e))
        return {"memory_id": "", "status": f"error: {e}", "domain": domain}


@tool(args_schema=QueryMemoryInput)
async def query_memory(
    query: str,
    top_k: int = 5,
    domain: str | None = None,
    min_importance: float = 0.0,
) -> dict[str, Any]:
    """
    Query the cross-domain vector memory using semantic search.
    Returns the top-k most similar entries, optionally filtered by domain
    and minimum importance score. Uses MemCollab temporal decay weighting.
    """
    logger.info(
        "memory_tools.query", query=query[:100], top_k=top_k, domain=domain
    )

    try:
        store = _get_vector_store()

        # Build metadata filter
        where_filter: dict[str, Any] = {}
        if domain:
            where_filter["domain"] = domain
        if min_importance > 0:
            where_filter["importance"] = {"$gte": str(min_importance)}

        results = await store.query(
            query_text=query,
            n_results=top_k,
            where=where_filter if where_filter else None,
        )

        entries = []
        for i, (doc, meta, score) in enumerate(
            zip(
                results.get("documents", [[]])[0],
                results.get("metadatas", [[]])[0],
                results.get("distances", [[]])[0],
            )
        ):
            entries.append(
                MemoryEntry(
                    id=results.get("ids", [[]])[0][i],
                    content=doc,
                    domain=meta.get("domain", "general"),
                    tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
                    importance=float(meta.get("importance", 0.5)),
                    similarity_score=1.0 - score,  # Convert distance to similarity
                    created_at=meta.get("created_at", ""),
                    metadata={
                        k: v
                        for k, v in meta.items()
                        if k not in ("domain", "tags", "importance", "created_at")
                    },
                )
            )

        result = QueryMemoryResult(
            query=query,
            results=entries,
            total_found=len(entries),
        )
        return result.model_dump()

    except Exception as e:
        logger.error("memory_tools.query.error", error=str(e))
        return {"query": query, "results": [], "total_found": 0, "error": str(e)}


@tool(args_schema=ForgetStaleInput)
async def forget_stale(
    max_age_days: int = 30,
    domain: str | None = None,
    min_importance_to_keep: float = 0.8,
) -> dict[str, Any]:
    """
    Remove stale memory entries older than max_age_days.
    Entries with importance >= min_importance_to_keep are preserved.
    This prevents overfitting to outdated knowledge (MemCollab principle).
    """
    logger.info(
        "memory_tools.forget_stale",
        max_age_days=max_age_days,
        domain=domain,
    )

    try:
        store = _get_vector_store()

        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_days * 86400)

        removed_count = await store.remove_stale(
            cutoff_timestamp=cutoff,
            domain=domain,
            min_importance_to_keep=min_importance_to_keep,
        )

        result = ForgetStaleResult(
            removed_count=removed_count,
            remaining_count=await store.count(domain=domain),
            domain=domain,
        )
        return result.model_dump()

    except Exception as e:
        logger.error("memory_tools.forget_stale.error", error=str(e))
        return {"removed_count": 0, "error": str(e)}


def get_memory_tools() -> list:
    """Return all memory-related LangChain tools."""
    return [store_memory, query_memory, forget_stale]
