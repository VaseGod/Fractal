"""
Fractal — Memory Module Tests
Tests for vector store and MemCollab architecture.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.memcollab import (
    BiasReport,
    MemCollabConfig,
    MemCollabManager,
    MemoryRecord,
)
from src.evaluation.scoring import EfficiencyScorer, EfficiencyMetrics


class TestMemoryRecord:
    """Test MemoryRecord model."""

    def test_defaults(self):
        record = MemoryRecord(content="Test memory entry")
        assert record.content == "Test memory entry"
        assert record.domain == "general"
        assert record.importance == 0.5
        assert record.access_count == 0
        assert record.decay_factor == 1.0

    def test_custom(self):
        record = MemoryRecord(
            content="DOM traversal strategy",
            domain="web_traversal",
            tags=["dom", "strategy"],
            importance=0.9,
            source_agent="meta_agent",
        )
        assert record.domain == "web_traversal"
        assert record.importance == 0.9
        assert len(record.tags) == 2


class TestMemCollabConfig:
    """Test MemCollab configuration."""

    def test_defaults(self):
        cfg = MemCollabConfig()
        assert cfg.decay_halflife_hours == 168.0  # 1 week
        assert cfg.min_importance_threshold == 0.3
        assert cfg.max_domain_concentration == 0.6
        assert cfg.merge_similarity_threshold == 0.85


class TestMemCollabDecay:
    """Test temporal decay computation."""

    def test_fresh_memory_has_full_decay(self):
        config = MemCollabConfig()
        manager = MemCollabManager.__new__(MemCollabManager)
        manager.config = config

        now = datetime.now(timezone.utc).isoformat()
        decay = manager.compute_decay(now)
        assert decay > 0.99

    def test_old_memory_decays(self):
        config = MemCollabConfig(decay_halflife_hours=24)
        manager = MemCollabManager.__new__(MemCollabManager)
        manager.config = config

        # 24 hours ago = half-life = ~0.5
        old = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        decay = manager.compute_decay(old)
        assert 0.45 < decay < 0.55

    def test_very_old_memory_near_zero(self):
        config = MemCollabConfig(decay_halflife_hours=24)
        manager = MemCollabManager.__new__(MemCollabManager)
        manager.config = config

        very_old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        decay = manager.compute_decay(very_old)
        assert decay < 0.05

    def test_invalid_timestamp_returns_default(self):
        config = MemCollabConfig()
        manager = MemCollabManager.__new__(MemCollabManager)
        manager.config = config

        decay = manager.compute_decay("invalid-date")
        assert decay == 0.5


class TestEfficiencyScorer:
    """Test efficiency scoring module."""

    def test_perfect_efficiency(self):
        scorer = EfficiencyScorer()
        metrics = scorer.score_task(
            actions_taken=1,
            task_solved=True,
            novel_observations=1,
        )
        assert metrics.normalized_score > 0.8
        assert metrics.action_count == 1

    def test_zero_actions(self):
        scorer = EfficiencyScorer()
        metrics = scorer.score_task(
            actions_taken=0,
            task_solved=True,
        )
        assert metrics.normalized_score == 1.0

    def test_unsolved_with_many_actions(self):
        scorer = EfficiencyScorer()
        metrics = scorer.score_task(
            actions_taken=50,
            task_solved=False,
        )
        assert metrics.normalized_score < 0.5

    def test_batch_scoring(self):
        scorer = EfficiencyScorer()
        task_metrics = [
            scorer.score_task(actions_taken=2, task_solved=True),
            scorer.score_task(actions_taken=10, task_solved=True),
            scorer.score_task(actions_taken=20, task_solved=False),
        ]
        report = scorer.score_batch(task_metrics)
        assert report.total_tasks == 3
        assert report.avg_normalized_score > 0
        assert report.total_actions == 32

    def test_comparison(self):
        scorer = EfficiencyScorer()
        baseline_metrics = [
            scorer.score_task(actions_taken=10, task_solved=True),
            scorer.score_task(actions_taken=15, task_solved=False),
        ]
        baseline = scorer.score_batch(baseline_metrics)

        improved_metrics = [
            scorer.score_task(actions_taken=3, task_solved=True),
            scorer.score_task(actions_taken=5, task_solved=True),
        ]
        improved = scorer.score_batch(improved_metrics)

        comparison = scorer.compare_runs(baseline, improved)
        assert comparison["score_delta"] > 0
        assert comparison["improved"] is True
