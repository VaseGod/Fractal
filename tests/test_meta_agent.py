"""
Fractal — Meta Agent Tests
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agents.meta_agent import (
    MetaAnalysisInput,
    PerformanceMetrics,
    FailurePattern,
    OptimizationProposal,
    MetaAnalysisResult,
)


class TestMetaAnalysisInput:
    """Test MetaAnalysisInput validation."""

    def test_defaults(self):
        inp = MetaAnalysisInput()
        assert inp.time_window_hours == 24
        assert inp.include_proposals is True
        assert inp.max_proposals == 5
        assert len(inp.analysis_id) > 0

    def test_custom_window(self):
        inp = MetaAnalysisInput(time_window_hours=168)
        assert inp.time_window_hours == 168

    def test_window_bounds(self):
        with pytest.raises(ValidationError):
            MetaAnalysisInput(time_window_hours=0)

        with pytest.raises(ValidationError):
            MetaAnalysisInput(time_window_hours=721)


class TestPerformanceMetrics:
    """Test PerformanceMetrics model."""

    def test_defaults(self):
        m = PerformanceMetrics()
        assert m.error_rate == 0.0
        assert m.total_executions == 0
        assert m.tool_efficiency == 0.0

    def test_populated(self):
        m = PerformanceMetrics(
            error_rate=0.15,
            avg_completion_time_ms=2500.0,
            tool_efficiency=0.85,
            total_executions=100,
        )
        assert m.error_rate == 0.15
        assert m.total_executions == 100


class TestFailurePattern:
    """Test FailurePattern model."""

    def test_creation(self):
        p = FailurePattern(
            pattern_type="failure",
            description="Recurring timeout in web extraction",
            evidence=["trace-1", "trace-2"],
            confidence=0.8,
        )
        assert p.pattern_type == "failure"
        assert len(p.evidence) == 2

    def test_frequency(self):
        p = FailurePattern(
            pattern_type="drift",
            description="Environmental drift detected",
            frequency=0.35,
            confidence=0.6,
        )
        assert p.frequency == 0.35


class TestOptimizationProposal:
    """Test OptimizationProposal model."""

    def test_creation(self):
        prop = OptimizationProposal(
            action="modify_config",
            target="task_agent.temperature",
            change_description="Reduce temperature from 0.2 to 0.1",
            expected_impact="5% reduction in hallucination rate",
            rollback_procedure="Revert to temperature=0.2",
            confidence=0.75,
        )
        assert prop.action == "modify_config"
        assert prop.requires_hitl is True
        assert len(prop.proposal_id) > 0


class TestMetaAnalysisResult:
    """Test MetaAnalysisResult model."""

    def test_empty_result(self):
        r = MetaAnalysisResult(analysis_id="test-123")
        assert r.analysis_id == "test-123"
        assert r.memory_entries_stored == 0
        assert len(r.patterns) == 0
        assert len(r.proposals) == 0

    def test_full_result(self):
        r = MetaAnalysisResult(
            analysis_id="test-456",
            metrics=PerformanceMetrics(error_rate=0.05, total_executions=50),
            patterns=[
                FailurePattern(
                    pattern_type="success",
                    description="Efficient DOM traversal strategy",
                    confidence=0.9,
                )
            ],
            proposals=[
                OptimizationProposal(
                    action="archive_strategy",
                    target="dom_traversal",
                    change_description="Archive efficient DOM traversal",
                    expected_impact="Reuse in future web tasks",
                    rollback_procedure="N/A",
                )
            ],
            memory_entries_stored=1,
            configs_archived=1,
        )
        assert r.metrics.error_rate == 0.05
        assert len(r.patterns) == 1
        assert len(r.proposals) == 1
        assert r.configs_archived == 1
