/**
 * Fractal — Evaluation Job (Trigger.dev)
 * Background job for running evaluation pipelines asynchronously.
 */

import { task } from "@trigger.dev/sdk/v3";
import { z } from "zod";

// ── Schema Definitions ──

const EvaluationPayload = z.object({
  evaluationId: z.string(),
  taskAgentTraceIds: z.array(z.string()).min(1),
  metrics: z
    .array(z.enum(["error_rate", "completion_time", "tool_efficiency", "hitl_rate"]))
    .default(["error_rate", "completion_time", "tool_efficiency"]),
  timeWindowHours: z.number().min(1).max(720).default(24),
});

const BenchmarkPayload = z.object({
  benchmarkId: z.string(),
  taskType: z.enum(["arc_agi_3", "spatial_logic", "dom_traversal", "custom"]),
  config: z.record(z.unknown()).default({}),
  maxIterations: z.number().min(1).max(1000).default(100),
});

// ── Job: Run Evaluation Pipeline ──

export const runEvaluation = task({
  id: "run-evaluation",
  retry: { maxAttempts: 1 },
  run: async (payload: z.infer<typeof EvaluationPayload>) => {
    const input = EvaluationPayload.parse(payload);
    const startTime = Date.now();

    // In production, this calls back to the Python orchestrator's
    // evaluation endpoint to run the Meta Agent analysis pipeline
    const orchestratorUrl =
      process.env.ORCHESTRATOR_URL || "http://orchestrator:8080";

    try {
      const response = await fetch(
        `${orchestratorUrl}/api/internal/evaluate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            evaluation_id: input.evaluationId,
            trace_ids: input.taskAgentTraceIds,
            metrics: input.metrics,
            time_window_hours: input.timeWindowHours,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(
          `Evaluation request failed: ${response.status} ${response.statusText}`
        );
      }

      const result = await response.json();

      return {
        evaluationId: input.evaluationId,
        status: "completed",
        result,
        executionTimeMs: Date.now() - startTime,
      };
    } catch (error) {
      return {
        evaluationId: input.evaluationId,
        status: "failed",
        error: error instanceof Error ? error.message : String(error),
        executionTimeMs: Date.now() - startTime,
      };
    }
  },
});

// ── Job: Run Benchmark Suite ──

export const runBenchmark = task({
  id: "run-benchmark",
  retry: { maxAttempts: 1 },
  run: async (payload: z.infer<typeof BenchmarkPayload>) => {
    const input = BenchmarkPayload.parse(payload);
    const startTime = Date.now();

    const orchestratorUrl =
      process.env.ORCHESTRATOR_URL || "http://orchestrator:8080";

    try {
      const response = await fetch(
        `${orchestratorUrl}/api/internal/benchmark`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            benchmark_id: input.benchmarkId,
            task_type: input.taskType,
            config: input.config,
            max_iterations: input.maxIterations,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(
          `Benchmark request failed: ${response.status} ${response.statusText}`
        );
      }

      const result = await response.json();

      return {
        benchmarkId: input.benchmarkId,
        taskType: input.taskType,
        status: "completed",
        result,
        executionTimeMs: Date.now() - startTime,
      };
    } catch (error) {
      return {
        benchmarkId: input.benchmarkId,
        taskType: input.taskType,
        status: "failed",
        error: error instanceof Error ? error.message : String(error),
        executionTimeMs: Date.now() - startTime,
      };
    }
  },
});
