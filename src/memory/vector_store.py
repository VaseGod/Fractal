"""
Fractal — Vector Store
ChromaDB-backed vector storage with async operations,
embedding generation, and metadata filtering.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any

import chromadb
import structlog
from chromadb.config import Settings
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class VectorStoreConfig(BaseModel):
    """Configuration for the vector store."""

    host: str = Field(default="localhost")
    port: int = Field(default=8100)
    collection_name: str = Field(default="fractal_memory")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    distance_metric: str = Field(default="cosine")


class VectorStore:
    """
    ChromaDB-based vector store for Fractal's cross-domain memory.

    Manages collections, embedding generation, similarity search,
    and metadata-filtered retrieval. Designed to work with the
    MemCollab architecture for temporal decay and bias mitigation.
    """

    def __init__(self, config: VectorStoreConfig | None = None):
        self.config = config or VectorStoreConfig(
            host=os.getenv("CHROMA_HOST", "localhost"),
            port=int(os.getenv("CHROMA_PORT", "8100")),
            collection_name=os.getenv("CHROMA_COLLECTION", "fractal_memory"),
        )

        # Initialize ChromaDB client
        try:
            self.client = chromadb.HttpClient(
                host=self.config.host,
                port=self.config.port,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False,
                ),
            )
            logger.info(
                "vector_store.connected",
                host=self.config.host,
                port=self.config.port,
            )
        except Exception as e:
            logger.warning(
                "vector_store.connection_failed",
                error=str(e),
                fallback="ephemeral",
            )
            # Fallback to ephemeral client for local dev
            self.client = chromadb.EphemeralClient()

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_metric},
        )

        logger.info(
            "vector_store.initialized",
            collection=self.config.collection_name,
            count=self.collection.count(),
        )

    async def add(
        self,
        content: str,
        metadata: dict[str, str] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """
        Add a document to the vector store.
        Embedding is generated automatically by ChromaDB.
        """
        doc_id = doc_id or str(uuid.uuid4())
        metadata = metadata or {}

        # Ensure timestamp is present
        if "created_at" not in metadata:
            metadata["created_at"] = datetime.now(timezone.utc).isoformat()

        self.collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[metadata],
        )

        logger.debug(
            "vector_store.add",
            doc_id=doc_id,
            content_length=len(content),
        )
        return doc_id

    async def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Query the vector store using semantic similarity.
        Returns documents, metadatas, distances, and IDs.
        """
        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(n_results, self.collection.count() or 1),
        }

        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        try:
            results = self.collection.query(**kwargs)
            logger.debug(
                "vector_store.query",
                query_length=len(query_text),
                results_count=len(results.get("documents", [[]])[0]),
            )
            return results
        except Exception as e:
            logger.error("vector_store.query_error", error=str(e))
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

    async def get(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Retrieve documents by ID or metadata filter."""
        kwargs: dict[str, Any] = {"limit": limit}
        if ids:
            kwargs["ids"] = ids
        if where:
            kwargs["where"] = where

        return self.collection.get(**kwargs)

    async def update(
        self,
        doc_id: str,
        content: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Update a document's content and/or metadata."""
        kwargs: dict[str, Any] = {"ids": [doc_id]}
        if content:
            kwargs["documents"] = [content]
        if metadata:
            kwargs["metadatas"] = [metadata]

        self.collection.update(**kwargs)

    async def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        self.collection.delete(ids=ids)
        logger.debug("vector_store.delete", count=len(ids))

    async def remove_stale(
        self,
        cutoff_timestamp: float,
        domain: str | None = None,
        min_importance_to_keep: float = 0.8,
    ) -> int:
        """
        Remove stale entries older than the cutoff timestamp.
        High-importance entries are preserved.
        """
        # Fetch all entries (paginated in production)
        where_filter: dict[str, Any] = {}
        if domain:
            where_filter["domain"] = domain

        results = self.collection.get(
            where=where_filter if where_filter else None,
            limit=10000,
        )

        ids_to_delete: list[str] = []

        for i, meta in enumerate(results.get("metadatas", [])):
            if meta is None:
                continue

            # Check importance — keep high-importance entries
            importance = float(meta.get("importance", "0.5"))
            if importance >= min_importance_to_keep:
                continue

            # Check age
            created_at_str = meta.get("created_at", "")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(
                        created_at_str
                    ).timestamp()
                    if created_at < cutoff_timestamp:
                        ids_to_delete.append(results["ids"][i])
                except ValueError:
                    continue

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            logger.info(
                "vector_store.remove_stale",
                removed=len(ids_to_delete),
                domain=domain,
            )

        return len(ids_to_delete)

    async def count(self, domain: str | None = None) -> int:
        """Count documents, optionally filtered by domain."""
        if domain:
            results = self.collection.get(
                where={"domain": domain}, limit=1
            )
            # ChromaDB doesn't have a count with filter, so we approximate
            results_full = self.collection.get(
                where={"domain": domain}, limit=100000
            )
            return len(results_full.get("ids", []))
        return self.collection.count()

    def reset(self) -> None:
        """Reset the collection (DANGEROUS — use only in tests)."""
        self.client.delete_collection(self.config.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_metric},
        )
        logger.warning("vector_store.reset", collection=self.config.collection_name)
