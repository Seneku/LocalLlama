import { describe, expect, test } from "bun:test";

import type { BenchmarkManager } from "../server/benchmark";
import { defaultProfiles } from "../server/defaultProfiles";
import { buildSweepPlan, deriveBestSettings, rankSweep, SweepManager, type EstimateFn } from "../server/sweep";
import type {
  BenchmarkRun,
  BenchmarkSettings,
  LlamaProfile,
  MemoryEstimate,
  SweepPlan
} from "../src/shared/types";

function makeProfile(overrides: Partial<LlamaProfile> = {}): LlamaProfile {
  return {
    ...defaultProfiles[0],
    id: "sweep-test",
    name: "Sweep test",
    modelPath: "C:\\models\\test.gguf",
    gpuLayers: 999,
    ...overrides
  };
}

const BLOCKS = 32;
const MAX_NGL = BLOCKS + 1;

function effNgl(gpuLayers: number): number {
  return Math.min(Math.max(0, gpuLayers), MAX_NGL);
}

/** Estimator stub: recommends 28 layers, says full offload is over budget. */
function stubEstimate(overrides: Partial<MemoryEstimate> = {}): EstimateFn {
  return async (profile) =>
    ({
      backend: "CUDA",
      fit: effNgl(profile.gpuLayers) >= MAX_NGL ? "over" : "fits",
      estimatedVramMiB: 8000,
      vramHeadroomMiB: 2000,
      recommendation: effNgl(profile.gpuLayers) === MAX_NGL ? { gpuLayers: 28 } : null,
      model: { blockCount: BLOCKS },
      ...overrides
    }) as unknown as MemoryEstimate;
}

describe("buildSweepPlan", () => {
  test("builds a bounded stage-wise plan for a GPU profile", async () => {
    const plan = await buildSweepPlan(makeProfile(), { estimate: stubEstimate() });

    const axes = plan.stages.map((stage) => stage.axis);
    expect(axes).toEqual(["baseline", "gpuLayers", "flashAttention", "ubatchSize", "batchSize"]);

    const gpuStage = plan.stages.find((stage) => stage.axis === "gpuLayers")!;
    // Current (full offload) is measured by the baseline; candidates come from the recommendation.
    const values = gpuStage.candidates.map((candidate) => candidate.profileOverrides.gpuLayers);
    expect(values).toEqual([28, 26, 24]);
    expect(gpuStage.candidates.every((candidate) => candidate.prunedReason === null)).toBe(true);

    // Bounded: baseline 1 + ngl 3 + fa 2 + ubatch 3 + batch 3 = 12 candidates.
    expect(plan.estimatedRuns).toBe(12);
    expect(plan.benchSettings.repetitions).toBe(3);
  });

  test("prunes gpu-layer candidates the estimator rejects", async () => {
    const estimate: EstimateFn = async (profile) =>
      ({
        backend: "CUDA",
        fit: effNgl(profile.gpuLayers) >= 28 ? "over" : "fits",
        estimatedVramMiB: 9000,
        vramHeadroomMiB: 2000,
        recommendation: effNgl(profile.gpuLayers) === MAX_NGL ? { gpuLayers: 28 } : null,
        model: { blockCount: BLOCKS }
      }) as unknown as MemoryEstimate;

    const plan = await buildSweepPlan(makeProfile(), { estimate });
    const gpuStage = plan.stages.find((stage) => stage.axis === "gpuLayers")!;
    const pruned = gpuStage.candidates.filter((candidate) => candidate.prunedReason !== null);
    expect(pruned.map((candidate) => candidate.profileOverrides.gpuLayers)).toEqual([28]);
    // Pruned candidates do not count toward the run estimate.
    expect(plan.estimatedRuns).toBe(11);
  });

  test("skips GPU stage on CPU backends and pinned flash attention", async () => {
    const plan = await buildSweepPlan(makeProfile({ backendMode: "cpu", flashAttention: "on" }), {
      estimate: stubEstimate({ backend: "CPU" } as Partial<MemoryEstimate>)
    });
    const axes = plan.stages.map((stage) => stage.axis);
    expect(axes).not.toContain("gpuLayers");
    expect(axes).not.toContain("flashAttention");
    expect(plan.notes.some((note) => note.includes("CPU backend"))).toBe(true);
    expect(plan.notes.some((note) => note.includes("pinned"))).toBe(true);
  });

  test("adds a q8_0 KV candidate only when VRAM headroom is tight", async () => {
    const roomy = await buildSweepPlan(makeProfile(), { estimate: stubEstimate() });
    expect(roomy.stages.map((stage) => stage.axis)).not.toContain("kvCache");

    const tight = await buildSweepPlan(makeProfile(), {
      estimate: stubEstimate({ vramHeadroomMiB: 800 } as Partial<MemoryEstimate>)
    });
    const kvStage = tight.stages.find((stage) => stage.axis === "kvCache")!;
    expect(kvStage.candidates[0].profileOverrides.kvCacheK).toBe("q8_0");
  });

  test("offers an f16 comparison when the profile already quantizes KV", async () => {
    const plan = await buildSweepPlan(makeProfile({ kvCacheK: "q8_0", kvCacheV: "q8_0" }), {
      estimate: stubEstimate()
    });
    const kvStage = plan.stages.find((stage) => stage.axis === "kvCache")!;
    expect(kvStage.candidates[0].profileOverrides.kvCacheK).toBe("f16");
  });

  test("quick mode shortens the benchmark settings", async () => {
    const plan = await buildSweepPlan(makeProfile(), { estimate: stubEstimate(), quick: true });
    expect(plan.benchSettings.repetitions).toBe(2);
    expect(plan.benchSettings.promptTokens).toBe(256);
  });
});

// ---- ranking ----

let runCounter = 0;

function makeRun(
  profile: LlamaProfile,
  settings: BenchmarkSettings,
  label: string,
  score: number,
  stddev = 0.1
): BenchmarkRun {
  runCounter += 1;
  return {
    id: `run-${runCounter}`,
    profileId: profile.id,
    profileName: profile.name,
    createdAt: "2026-07-09T12:00:00.000Z",
    completedAt: "2026-07-09T12:00:30.000Z",
    status: "completed",
    exitCode: 0,
    signal: null,
    backend: "CUDA",
    settings,
    command: {
      executable: "llama-bench",
      args: [],
      display: "",
      backend: "CUDA",
      benchmarkExists: true,
      modelExists: true,
      warnings: []
    },
    profile: {
      name: profile.name,
      modelPath: profile.modelPath,
      backendMode: profile.backendMode,
      contextSize: profile.contextSize,
      threadsMode: profile.threadsMode,
      threads: profile.threads,
      gpuLayers: profile.gpuLayers,
      kvCacheK: profile.kvCacheK,
      kvCacheV: profile.kvCacheV,
      jinja: profile.jinja,
      reasoning: profile.reasoning,
      parallelSlots: profile.parallelSlots,
      speculativeEnabled: profile.speculative.enabled,
      speculativeType: profile.speculative.type
    },
    rows: [],
    metrics: {
      promptTokensPerSecond: score,
      generationTokensPerSecond: score,
      generationMsPerToken: 1000 / score,
      promptStddev: stddev,
      generationStddev: stddev,
      totalSeconds: 30,
      score
    },
    env: null,
    sweepId: "sweep-x",
    sweepLabel: label,
    stdout: "",
    stderr: "",
    error: null
  };
}

const benchDefaults: BenchmarkSettings = {
  promptTokens: 512,
  generationTokens: 128,
  repetitions: 3,
  batchSize: 2048,
  ubatchSize: 512,
  noWarmup: false,
  flashAttention: "auto"
};

describe("rankSweep", () => {
  test("picks the clear top scorer and flags nothing as tied", () => {
    const profile = makeProfile();
    const baseline = makeRun(profile, benchDefaults, "baseline (current profile)", 90);
    const better = makeRun({ ...profile, gpuLayers: 26 }, benchDefaults, "gpu layers 26", 100);

    const ranking = rankSweep([baseline, better]);
    expect(ranking.winnerRunId).toBe(better.id);
    expect(ranking.baselineRunId).toBe(baseline.id);
    expect(ranking.ranked[0].runId).toBe(better.id);
    expect(ranking.ranked[1].withinNoiseOfBest).toBe(false);
  });

  test("prefers the simpler config when scores are statistically tied", () => {
    const profile = makeProfile();
    // Tuned config scores nominally higher but within noise of the default config.
    const noisy = 3; // stddev → sigma ≈ score * 0.5 * sqrt(2) * (3/100) ≈ 2.1
    const tuned = makeRun(profile, { ...benchDefaults, ubatchSize: 256, flashAttention: "on" }, "micro-batch 256", 101, noisy);
    const plain = makeRun(profile, benchDefaults, "baseline (current profile)", 100, noisy);

    const ranking = rankSweep([tuned, plain]);
    expect(ranking.ranked[0].runId).toBe(tuned.id); // still ranked by score
    expect(ranking.winnerRunId).toBe(plain.id); // but the simpler config wins
    expect(ranking.ranked[1].withinNoiseOfBest).toBe(true);
    expect(ranking.notes.some((note) => note.includes("statistically tied"))).toBe(true);
  });

  test("ignores failed runs and runs without a score", () => {
    const profile = makeProfile();
    const good = makeRun(profile, benchDefaults, "baseline (current profile)", 50);
    const failed = { ...makeRun(profile, benchDefaults, "gpu layers 26", 999), status: "failed" as const };

    const ranking = rankSweep([good, failed]);
    expect(ranking.ranked).toHaveLength(1);
    expect(ranking.winnerRunId).toBe(good.id);
  });
});

// ---- orchestration ----

/**
 * Fake BenchmarkManager: completes every run instantly with a deterministic
 * score that peaks at 26 GPU layers, +1 for flash attention on, +3 for
 * micro-batch 256.
 */
class FakeBenchmark {
  executed: Array<{ profile: LlamaProfile; settings: BenchmarkSettings; label: string }> = [];
  cancelNext = false;

  async start(
    profile: LlamaProfile,
    settings: BenchmarkSettings,
    extras: { sweepId?: string; sweepLabel?: string; onComplete?: (run: BenchmarkRun) => void }
  ): Promise<BenchmarkRun> {
    this.executed.push({ profile, settings, label: extras.sweepLabel ?? "?" });
    const score = 100 - Math.abs(effNgl(profile.gpuLayers) - 26) * 5 +
      (settings.flashAttention === "on" ? 1 : 0) +
      (settings.ubatchSize === 256 ? 3 : 0);
    const run = makeRun(profile, settings, extras.sweepLabel ?? "?", score);
    if (this.cancelNext) {
      run.status = "cancelled";
    }
    setTimeout(() => extras.onComplete?.(run), 0);
    return run;
  }

  getStatus() {
    return { state: "idle" as const, activeRunId: null, profileId: null, profileName: null, startedAt: null, command: null };
  }

  async stop() {
    this.cancelNext = true;
    return this.getStatus();
  }
}

async function waitForIdle(sweep: SweepManager): Promise<void> {
  for (let i = 0; i < 1000 && sweep.getStatus().state === "running"; i++) {
    await new Promise((resolve) => setTimeout(resolve, 1));
  }
}

describe("SweepManager", () => {
  test("runs the plan stage-wise, dedupes identical configs, and applies the winner", async () => {
    const profile = makeProfile();
    const fake = new FakeBenchmark();
    const sweep = new SweepManager(fake as unknown as BenchmarkManager, { delayMs: 0 });
    const plan = await buildSweepPlan(profile, { estimate: stubEstimate() });

    sweep.start(profile, plan);
    expect(() => sweep.start(profile, plan)).toThrow("already running");
    await waitForIdle(sweep);

    const result = sweep.getLastResult()!;
    expect(result.status).toBe("completed");

    // 12 candidates, but "micro-batch 512" and "batch 2048" duplicate earlier
    // configs and are reused instead of re-run.
    expect(fake.executed).toHaveLength(10);

    // Winner: ngl 26 + fa on + ubatch 256 (score 104).
    const winner = result.ranked[0];
    expect(result.winnerRunId).toBe(winner.runId);
    expect(winner.score).toBe(104);
    expect(result.bestSettings.gpuLayers).toBe(26);
    expect(result.bestSettings.flashAttention).toBe("on");
    expect(result.bestSettings.ubatchSize).toBe(256);
    expect(result.bestSettings.batchSize).toBe(0); // 2048 = llama.cpp default → unset
    expect(result.baselineRunId).not.toBeNull();
    expect(sweep.getStatus().state).toBe("idle");
  });

  test("stop cancels the sweep and reports partial results", async () => {
    const profile = makeProfile();
    const fake = new FakeBenchmark();
    const sweep = new SweepManager(fake as unknown as BenchmarkManager, { delayMs: 5 });
    const plan = await buildSweepPlan(profile, { estimate: stubEstimate() });

    sweep.start(profile, plan);
    // Let the baseline complete, then cancel.
    await new Promise((resolve) => setTimeout(resolve, 8));
    await sweep.stop();
    await waitForIdle(sweep);

    const result = sweep.getLastResult()!;
    expect(result.status).toBe("cancelled");
    expect(fake.executed.length).toBeLessThan(10);
    expect(sweep.getStatus().state).toBe("idle");
  });
});
