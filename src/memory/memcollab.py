"""
Fractal — MemCollab Architecture
Cross-domain collaborative memory with temporal decay,
bias detection/mitigation, and strategy merging.
"""

from __future__ import annotations

import math
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from pydantic import BaseModel, Field

from src.memory.vector_store import VectorStore

logger = structlog.get_logger(__name__)


class MemoryRecord(BaseModel):
    """A single memory record with MemCollab metadata."""

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    domain: str = "general"
    tags: list[str] = Field(default_factory=list)
    importance: float = 0.5
    access_count: int = 0
    last_accessed: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source_agent: str = "task_agent"
    decay_factor: float = 1.0  # 1.0 = fresh, approaches 0 over time


class MemCollabConfig(BaseModel):
    """Configuration for MemCollab behavior."""

    decay_halflife_hours: float = Field(
        default=168.0,  # 1 week
        description="Half-life for temporal decay (hours)",
    )
    min_importance_threshold: float = Field(
        default=0.3,
        description="Minimum importance to keep in active memory",
    )
    bias_detection_window: int = Field(
        default=100,
        description="Number of recent queries to analyze for bias",
    )
    max_domain_concentration: float = Field(
        default=0.6,
        description="Max fraction of results from a single domain before bias alert",
    )
    merge_similarity_threshold: float = Field(
        default=0.85,
        description="Similarity threshold for merging duplicate memories",
    )


class BiasReport(BaseModel):
    """Report on detected bias in memory retrieval."""

    is_biased: bool = False
    dominant_domain: str | None = None
    domain_distribution: dict[str, float] = Field(default_factory=dict)
    recommendation: str = ""


class MergeResult(BaseModel):
    """Result from merging similar memories."""

    merged_count: int = 0
    new_record_ids: list[str] = Field(default_factory=list)
    removed_record_ids: list[str] = Field(default_factory=list)


class MemCollabManager:
    """
    MemCollab Architecture Implementation.

    Provides cross-domain memory management with:
    - Temporal decay weighting (memories lose relevance over time)
    - Bias detection and mitigation (prevents domain overfitting)
    - Collaborative memory merging between agents
    - Strategy retrieval with context-aware ranking
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        config: MemCollabConfig | None = None,
    ):
        self.config = config or MemCollabConfig()
        self.store = vector_store or VectorStore()
        self._query_history: list[dict[str, Any]] = []

        logger.info(
            "memcollab.initialized",
            decay_halflife_hours=self.config.decay_halflife_hours,
            bias_threshold=self.config.max_domain_concentration,
        )

    def compute_decay(self, created_at: str) -> float:
        """
        Compute temporal decay factor using exponential decay.
        Returns a value between 0 (completely decayed) and 1 (fresh).
        """
        try:
            created = datetime.fromisoformat(created_at)
            now = datetime.now(timezone.utc)
            hours_elapsed = (now - created).total_seconds() / 3600

            # Exponential decay: factor = 2^(-t / halflife)
            decay = math.pow(2, -hours_elapsed / self.config.decay_halflife_hours)
            return max(0.01, min(1.0, decay))  # Clamp to [0.01, 1.0]

        except (ValueError, TypeError):
            return 0.5  # Default for unparseable timestamps

    async def store_memory(
        self,
        content: str,
        domain: str = "general",
        tags: list[str] | None = None,
        importance: float = 0.5,
        source_agent: str = "task_agent",
    ) -> str:
        """
        Store a memory with MemCollab metadata.
        Checks for duplicates before storing.
        """
        tags = tags or []

        # Check for near-duplicates
        existing = await self.store.query(
            query_text=content,
            n_results=3,
        )

        for i, doc_list in enumerate(existing.get("documents", [[]])):
            for j, doc in enumerate(doc_list):
                distance = existing.get("distances", [[]])[i][j]
                similarity = 1.0 - distance
                if similarity >= self.config.merge_similarity_threshold:
                    logger.info(
                        "memcollab.duplicate_detected",
                        similarity=similarity,
                        existing_id=existing["ids"][i][j],
                    )
                    # Update access count of existing entry instead
                    existing_meta = existing.get("metadatas", [[]])[i][j]
                    access_count = int(existing_meta.get("access_count", "0")) + 1
                    await self.store.update(
                        doc_id=existing["ids"][i][j],
                        metadata={
                            **existing_meta,
                            "access_count": str(access_count),
                            "last_accessed": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    return existing["ids"][i][j]

        # No duplicate — store new entry
        record = MemoryRecord(
            content=content,
            domain=domain,
            tags=tags,
            importance=importance,
            source_agent=source_agent,
        )

        doc_id = await self.store.add(
            content=content,
            metadata={
                "domain": domain,
                "tags": ",".join(tags),
                "importance": str(importance),
                "source_agent": source_agent,
                "access_count": "0",
                "last_accessed": record.last_accessed,
                "created_at": record.created_at,
            },
            doc_id=record.record_id,
        )

        return doc_id

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        domain: str | None = None,
        apply_decay: bool = True,
        apply_bias_correction: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Retrieve memories with MemCollab ranking.

        Applies temporal decay weighting and bias correction
        to produce a balanced, recency-aware result set.
        """
        # Fetch more than requested to allow for re-ranking
        fetch_k = min(top_k * 3, 50)

        where_filter = {"domain": domain} if domain else None

        raw_results = await self.store.query(
            query_text=query,
            n_results=fetch_k,
            where=where_filter,
        )

        # Build scored results
        scored_results: list[dict[str, Any]] = []

        for i, doc in enumerate(raw_results.get("documents", [[]])[0]):
            meta = raw_results.get("metadatas", [[]])[0][i]
            distance = raw_results.get("distances", [[]])[0][i]
            doc_id = raw_results.get("ids", [[]])[0][i]

            similarity = 1.0 - distance
            importance = float(meta.get("importance", "0.5"))

            # Apply temporal decay
            decay = 1.0
            if apply_decay:
                decay = self.compute_decay(meta.get("created_at", ""))

            # Composite score: similarity * importance * decay
            composite_score = similarity * (0.5 + 0.5 * importance) * decay

            scored_results.append(
                {
                    "id": doc_id,
                    "content": doc,
                    "domain": meta.get("domain", "general"),
                    "tags": meta.get("tags", "").split(",") if meta.get("tags") else [],
                    "importance": importance,
                    "similarity": similarity,
                    "decay_factor": decay,
                    "composite_score": composite_score,
                    "source_agent": meta.get("source_agent", "unknown"),
                    "created_at": meta.get("created_at", ""),
                    "access_count": int(meta.get("access_count", "0")),
                }
            )

        # Sort by composite score
        scored_results.sort(key=lambda x: x["composite_score"], reverse=True)

        # Apply bias correction
        if apply_bias_correction and len(scored_results) > 3:
            scored_results = self._apply_bias_correction(scored_results, top_k)

        # Track query for bias detection
        result_domains = [r["domain"] for r in scored_results[:top_k]]
        self._query_history.append(
            {
                "query": query[:100],
                "domains": result_domains,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Trim history
        if len(self._query_history) > self.config.bias_detection_window:
            self._query_history = self._query_history[
                -self.config.bias_detection_window :
            ]

        return scored_results[:top_k]

    def _apply_bias_correction(
        self, results: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """
        Apply diversity-based bias correction.
        Ensures no single domain dominates the results.
        """
        max_per_domain = max(
            2, int(top_k * self.config.max_domain_concentration)
        )

        corrected: list[dict[str, Any]] = []
        domain_counts: dict[str, int] = {}

        for result in results:
            domain = result["domain"]
            count = domain_counts.get(domain, 0)

            if count < max_per_domain:
                corrected.append(result)
                domain_counts[domain] = count + 1

            if len(corrected) >= top_k:
                break

        # If we don't have enough, fill from remaining
        if len(corrected) < top_k:
            remaining = [r for r in results if r not in corrected]
            corrected.extend(remaining[: top_k - len(corrected)])

        return corrected

    async def detect_bias(self) -> BiasReport:
        """
        Analyze recent query history for retrieval bias.
        Returns a report on domain concentration patterns.
        """
        if len(self._query_history) < 10:
            return BiasReport(
                recommendation="Insufficient query history for bias analysis."
            )

        # Count domain appearances in results
        domain_counts: dict[str, int] = {}
        total = 0

        for entry in self._query_history:
            for domain in entry.get("domains", []):
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                total += 1

        if total == 0:
            return BiasReport()

        # Compute distribution
        distribution = {d: c / total for d, c in domain_counts.items()}

        # Check for dominant domain
        max_domain = max(distribution, key=distribution.get)  # type: ignore
        max_concentration = distribution[max_domain]

        is_biased = max_concentration > self.config.max_domain_concentration

        recommendation = ""
        if is_biased:
            recommendation = (
                f"Domain '{max_domain}' accounts for {max_concentration:.1%} "
                f"of retrieval results (threshold: "
                f"{self.config.max_domain_concentration:.1%}). "
                f"Consider diversifying memory entries or adjusting query strategies."
            )

        return BiasReport(
            is_biased=is_biased,
            dominant_domain=max_domain if is_biased else None,
            domain_distribution=distribution,
            recommendation=recommendation,
        )

    async def merge_similar(self, domain: str | None = None) -> MergeResult:
        """
        Find and merge near-duplicate memories within a domain.
        Keeps the highest-importance version and combines metadata.
        """
        where_filter = {"domain": domain} if domain else None

        all_docs = await self.store.get(where=where_filter, limit=1000)

        ids = all_docs.get("ids", [])
        documents = all_docs.get("documents", [])
        metadatas = all_docs.get("metadatas", [])

        if len(ids) < 2:
            return MergeResult()

        merged_ids: set[str] = set()
        new_ids: list[str] = []
        removed_ids: list[str] = []

        for i in range(len(ids)):
            if ids[i] in merged_ids:
                continue

            for j in range(i + 1, len(ids)):
                if ids[j] in merged_ids:
                    continue

                # Check similarity
                results = await self.store.query(
                    query_text=documents[i],
                    n_results=2,
                )

                for k, result_id in enumerate(results.get("ids", [[]])[0]):
                    if result_id == ids[j]:
                        distance = results.get("distances", [[]])[0][k]
                        similarity = 1.0 - distance

                        if similarity >= self.config.merge_similarity_threshold:
                            # Merge: keep higher importance version
                            imp_i = float(
                                metadatas[i].get("importance", "0.5")
                            )
                            imp_j = float(
                                metadatas[j].get("importance", "0.5")
                            )

                            keep_idx = i if imp_i >= imp_j else j
                            remove_idx = j if keep_idx == i else i

                            await self.store.delete([ids[remove_idx]])
                            merged_ids.add(ids[remove_idx])
                            removed_ids.append(ids[remove_idx])

                            logger.info(
                                "memcollab.merged",
                                kept=ids[keep_idx],
                                removed=ids[remove_idx],
                                similarity=similarity,
                            )

        return MergeResult(
            merged_count=len(removed_ids),
            new_record_ids=new_ids,
            removed_record_ids=removed_ids,
        )

    async def decay_sweep(self) -> int:
        """
        Run a decay sweep: remove entries whose effective importance
        (importance * decay) falls below the minimum threshold.
        """
        all_docs = await self.store.get(limit=10000)

        ids_to_remove: list[str] = []

        for i, meta in enumerate(all_docs.get("metadatas", [])):
            if meta is None:
                continue

            importance = float(meta.get("importance", "0.5"))
            decay = self.compute_decay(meta.get("created_at", ""))
            effective_importance = importance * decay

            if effective_importance < self.config.min_importance_threshold:
                ids_to_remove.append(all_docs["ids"][i])

        if ids_to_remove:
            await self.store.delete(ids_to_remove)
            logger.info(
                "memcollab.decay_sweep",
                removed=len(ids_to_remove),
            )

        return len(ids_to_remove)
