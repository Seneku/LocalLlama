// One-click "Optimize" sweep: a bounded, stage-wise greedy search over the
// llama-bench-able parameter space. Each stage tunes one axis while carrying
// the best configuration found so far — never a cartesian grid — so a full
// sweep stays around 10–15 executed runs. Candidates that the VRAM estimator
// says cannot fit are pre-pruned and never executed. Plan building and
// ranking are pure(ish) and unit-tested; SweepManager wires them to the
// existing single-run BenchmarkManager.
import { defaultBenchmarkSettings, type BenchmarkManager } from "./benchmark";
import { estimateProfileMemory } from "./estimate";
import { normalizeProfile } from "./normalize";
import { getDefaultThreads } from "./paths";
import type {
  BenchmarkRun,
  BenchmarkSettings,
  LlamaProfile,
  MemoryEstimate,
  SweepAxisId,
  SweepCandidate,
  SweepPlan,
  SweepRanked,
  SweepResult,
  SweepStage,
  SweepStatus
} from "../src/shared/types";

export type EstimateFn = (profile: LlamaProfile) => Promise<MemoryEstimate>;

export interface BuildSweepPlanOptions {
  estimate?: EstimateFn;
  /** Axes to include; defaults to every axis except "threads" (rarely matters on full GPU offload). */
  axes?: SweepAxisId[];
  /** Quick mode: 2 repetitions and shorter prompts — faster, noisier. */
  quick?: boolean;
}

const DEFAULT_AXES: SweepAxisId[] = ["gpuLayers", "flashAttention", "ubatchSize", "batchSize", "kvCache"];

// Candidate grids. ubatch is tuned before batch (bigger effect on VRAM and
// prompt speed); batch candidates below the tuned ubatch are skipped at
// execution time (llama.cpp requires ub <= b).
const UBATCH_CANDIDATES = [128, 256, 512];
const BATCH_CANDIDATES = [512, 1024, 2048];

function effectiveGpuLayers(gpuLayers: number, maxGpuLayers: number | null): number {
  const floored = Math.max(0, Math.floor(gpuLayers));
  return maxGpuLayers !== null ? Math.min(floored, maxGpuLayers) : floored;
}

export async function buildSweepPlan(profile: LlamaProfile, options: BuildSweepPlanOptions = {}): Promise<SweepPlan> {
  const estimate = options.estimate ?? ((p: LlamaProfile) => estimateProfileMemory(p));
  const axes = options.axes ?? DEFAULT_AXES;
  const notes: string[] = [];
  const stages: SweepStage[] = [];

  const benchSettings: BenchmarkSettings = options.quick
    ? { ...defaultBenchmarkSettings, promptTokens: 256, generationTokens: 64, repetitions: 2 }
    : { ...defaultBenchmarkSettings };
  if (options.quick) {
    notes.push("Quick mode: 2 repetitions with shorter prompts — results are noisier than a full sweep.");
  }

  stages.push({
    axis: "baseline",
    title: "Baseline",
    candidates: [
      { axis: "baseline", label: "baseline (current profile)", settings: {}, profileOverrides: {}, prunedReason: null }
    ]
  });

  const baseEstimate = await estimate(profile);
  const usesGpu = baseEstimate.backend !== "CPU";
  const blockCount = baseEstimate.model.blockCount;
  const maxGpuLayers = blockCount ? blockCount + 1 : null;

  if (axes.includes("gpuLayers") && usesGpu && maxGpuLayers !== null) {
    const current = effectiveGpuLayers(profile.gpuLayers, maxGpuLayers);
    const recommended = baseEstimate.recommendation?.gpuLayers ?? null;
    const values = new Set<number>();
    if (recommended !== null) {
      values.add(recommended);
      if (recommended - 2 > 0) values.add(recommended - 2);
      if (recommended - 4 > 0) values.add(recommended - 4);
    }
    values.add(maxGpuLayers); // full offload — pruned below if it cannot fit
    values.delete(current); // the baseline already measures the current value

    const candidates: SweepCandidate[] = [];
    for (const value of [...values].sort((a, b) => b - a)) {
      let prunedReason: string | null = null;
      try {
        const fit = await estimate({ ...profile, gpuLayers: value });
        if (fit.fit === "over") {
          prunedReason = `estimated ${fit.estimatedVramMiB} MiB VRAM exceeds the available budget`;
        }
      } catch {
        // Estimator failure (e.g. unreadable model) — keep the candidate.
      }
      candidates.push({
        axis: "gpuLayers",
        label: value >= maxGpuLayers ? `gpu layers ${value} (full offload)` : `gpu layers ${value}`,
        settings: {},
        profileOverrides: { gpuLayers: value },
        prunedReason
      });
    }
    if (candidates.length > 0) {
      stages.push({ axis: "gpuLayers", title: "GPU layers", candidates });
    }
  } else if (axes.includes("gpuLayers") && !usesGpu) {
    notes.push("CPU backend: the GPU-layers stage is skipped.");
  }

  if (axes.includes("flashAttention")) {
    if (profile.flashAttention === "auto") {
      stages.push({
        axis: "flashAttention",
        title: "Flash attention",
        candidates: (["on", "off"] as const).map((mode) => ({
          axis: "flashAttention",
          label: `flash attention ${mode}`,
          settings: { flashAttention: mode },
          profileOverrides: { flashAttention: mode },
          prunedReason: null
        }))
      });
    } else {
      notes.push(`Flash attention is pinned to "${profile.flashAttention}" in the profile; stage skipped.`);
    }
  }

  if (axes.includes("ubatchSize")) {
    stages.push({
      axis: "ubatchSize",
      title: "Micro-batch size",
      candidates: UBATCH_CANDIDATES.map((value) => ({
        axis: "ubatchSize" as const,
        label: `micro-batch ${value}`,
        settings: { ubatchSize: value },
        profileOverrides: { ubatchSize: value },
        prunedReason: null
      }))
    });
  }

  if (axes.includes("batchSize")) {
    stages.push({
      axis: "batchSize",
      title: "Batch size",
      candidates: BATCH_CANDIDATES.map((value) => ({
        axis: "batchSize" as const,
        label: `batch ${value}`,
        settings: { batchSize: value },
        profileOverrides: { batchSize: value },
        prunedReason: null
      }))
    });
  }

  if (axes.includes("kvCache")) {
    const quantized = profile.kvCacheK === "q8_0" && profile.kvCacheV === "q8_0";
    const headroom = baseEstimate.vramHeadroomMiB;
    if (quantized) {
      // Already q8: measure whether f16 would be faster (pruned if it cannot fit).
      let prunedReason: string | null = null;
      try {
        const fit = await estimate({ ...profile, kvCacheK: "f16" as const, kvCacheV: "f16" as const });
        if (fit.fit === "over") {
          prunedReason = "f16 KV cache would not fit VRAM";
        }
      } catch {
        // Keep the candidate.
      }
      stages.push({
        axis: "kvCache",
        title: "KV cache quantization",
        candidates: [
          {
            axis: "kvCache",
            label: "kv cache f16",
            settings: {},
            profileOverrides: { kvCacheK: "f16", kvCacheV: "f16" },
            prunedReason
          }
        ]
      });
    } else if (usesGpu && headroom !== null && headroom < 1536) {
      stages.push({
        axis: "kvCache",
        title: "KV cache quantization",
        candidates: [
          {
            axis: "kvCache",
            label: "kv cache q8_0",
            settings: {},
            profileOverrides: { kvCacheK: "q8_0", kvCacheV: "q8_0" },
            prunedReason: null
          }
        ]
      });
      notes.push("VRAM headroom is under 1.5 GiB, so a q8_0 KV-cache candidate is included.");
    }
  }

  if (axes.includes("threads") && profile.threadsMode === "auto") {
    // "auto" uses all logical cores; the interesting alternatives are the
    // physical-core estimate (logical/2 with SMT) and all-but-one.
    const logical = getDefaultThreads();
    const values = [...new Set([Math.max(1, Math.floor(logical / 2)), Math.max(1, logical - 1)])].filter(
      (value) => value !== logical
    );
    if (values.length > 0) {
      stages.push({
        axis: "threads",
        title: "Threads",
        candidates: values.map((value) => ({
          axis: "threads" as const,
          label: `threads ${value}`,
          settings: {},
          profileOverrides: { threadsMode: "manual" as const, threads: value },
          prunedReason: null
        }))
      });
    }
  }

  const estimatedRuns = stages.reduce(
    (sum, stage) => sum + stage.candidates.filter((candidate) => !candidate.prunedReason).length,
    0
  );

  return {
    profileId: profile.id,
    profileName: profile.name,
    stages,
    benchSettings,
    maxGpuLayers,
    estimatedRuns,
    notes
  };
}

// ---- ranking ----

interface ScoredRun {
  run: BenchmarkRun;
  score: number;
  sigma: number;
}

function scoreSigma(run: BenchmarkRun): number | null {
  const { promptTokensPerSecond: pp, generationTokensPerSecond: tg, promptStddev, generationStddev, score } = run.metrics;
  if (score === null) {
    return null;
  }
  // score = sqrt(pp*tg) → relStd(score) ≈ ½·sqrt(relStd(pp)² + relStd(tg)²).
  // Missing stddevs fall back to a 2% floor so single-rep runs still rank.
  const relPp = pp && promptStddev !== null ? promptStddev / pp : 0.02;
  const relTg = tg && generationStddev !== null ? generationStddev / tg : 0.02;
  return score * 0.5 * Math.sqrt(relPp * relPp + relTg * relTg);
}

/** Count how far a run's config strays from llama.cpp defaults — the tie-breaker prefers simpler configs. */
function deviationCount(run: BenchmarkRun): number {
  let count = 0;
  if (run.settings.batchSize !== defaultBenchmarkSettings.batchSize) count += 1;
  if (run.settings.ubatchSize !== defaultBenchmarkSettings.ubatchSize) count += 1;
  if (run.settings.flashAttention !== "auto") count += 1;
  if (run.profile.kvCacheK || run.profile.kvCacheV) count += 1;
  if (run.profile.threadsMode === "manual") count += 1;
  return count;
}

export interface SweepRanking {
  ranked: SweepRanked[];
  winnerRunId: string | null;
  baselineRunId: string | null;
  notes: string[];
}

export function rankSweep(runs: BenchmarkRun[]): SweepRanking {
  const notes: string[] = [];
  const scored: ScoredRun[] = [];
  for (const run of runs) {
    if (run.status !== "completed" || run.metrics.score === null) {
      continue;
    }
    scored.push({ run, score: run.metrics.score, sigma: scoreSigma(run) ?? run.metrics.score * 0.02 });
  }
  scored.sort((a, b) => b.score - a.score);

  const baselineRunId = runs.find((run) => run.sweepLabel?.startsWith("baseline"))?.id ?? null;
  if (scored.length === 0) {
    return { ranked: [], winnerRunId: null, baselineRunId, notes: ["No completed runs produced a score."] };
  }

  const top = scored[0];
  const withinNoise = (entry: ScoredRun) =>
    top.score - entry.score < 2 * Math.sqrt(top.sigma * top.sigma + entry.sigma * entry.sigma);

  // Among runs statistically tied with the top scorer, prefer the config with
  // the fewest deviations from llama.cpp defaults (simpler + usually cooler).
  const tied = scored.filter(withinNoise);
  let winner = top;
  if (tied.length > 1) {
    winner = tied.reduce((best, entry) => (deviationCount(entry.run) < deviationCount(best.run) ? entry : best), top);
    if (winner !== top) {
      notes.push(
        `"${winner.run.sweepLabel ?? winner.run.id}" is statistically tied with the top scorer; picked it for having a simpler configuration.`
      );
    }
  }

  const ranked: SweepRanked[] = scored.map((entry) => ({
    runId: entry.run.id,
    label: entry.run.sweepLabel ?? entry.run.id,
    score: entry.score,
    scoreStddev: entry.sigma,
    withinNoiseOfBest: entry !== top && withinNoise(entry)
  }));

  return { ranked, winnerRunId: winner.run.id, baselineRunId, notes };
}

/** Map a winning run's stored config back onto profile fields. Values equal to llama.cpp defaults map to "unset". */
export function deriveBestSettings(run: BenchmarkRun): Partial<LlamaProfile> {
  return {
    gpuLayers: run.profile.gpuLayers,
    threadsMode: run.profile.threadsMode,
    threads: run.profile.threads,
    kvCacheK: run.profile.kvCacheK,
    kvCacheV: run.profile.kvCacheV,
    batchSize: run.settings.batchSize === defaultBenchmarkSettings.batchSize ? 0 : run.settings.batchSize,
    ubatchSize: run.settings.ubatchSize === defaultBenchmarkSettings.ubatchSize ? 0 : run.settings.ubatchSize,
    flashAttention: run.settings.flashAttention
  };
}

// ---- orchestration ----

export interface SweepManagerDeps {
  /** Inter-run pause; Windows needs a beat for the previous process to release VRAM. */
  delayMs?: number;
  now?: () => Date;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function configKey(profile: LlamaProfile, settings: BenchmarkSettings, maxGpuLayers: number | null): string {
  return [
    `ngl:${effectiveGpuLayers(profile.gpuLayers, maxGpuLayers)}`,
    `fa:${settings.flashAttention}`,
    `b:${settings.batchSize}`,
    `ub:${settings.ubatchSize}`,
    `ctk:${profile.kvCacheK || "f16"}`,
    `ctv:${profile.kvCacheV || "f16"}`,
    `t:${profile.threadsMode}:${profile.threads}`
  ].join("|");
}

export class SweepManager {
  private sweepId: string | null = null;
  private profileId: string | null = null;
  private profileName: string | null = null;
  private startedAt: string | null = null;
  private totalRuns = 0;
  private completedRuns = 0;
  private currentCandidate: string | null = null;
  private cancelled = false;
  private running = false;
  private lastResult: SweepResult | null = null;
  private readonly delayMs: number;

  constructor(private readonly benchmark: BenchmarkManager, deps: SweepManagerDeps = {}) {
    this.delayMs = deps.delayMs ?? 3000;
  }

  getStatus(): SweepStatus {
    return {
      state: this.running ? "running" : "idle",
      sweepId: this.sweepId,
      profileId: this.profileId,
      profileName: this.profileName,
      completedRuns: this.completedRuns,
      totalRuns: this.totalRuns,
      currentCandidate: this.currentCandidate,
      startedAt: this.startedAt
    };
  }

  getLastResult(): SweepResult | null {
    return this.lastResult;
  }

  start(profile: LlamaProfile, plan: SweepPlan): SweepStatus {
    if (this.running) {
      throw new Error("A sweep is already running.");
    }
    this.running = true;
    this.cancelled = false;
    this.sweepId = `sweep-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    this.profileId = profile.id;
    this.profileName = profile.name;
    this.startedAt = new Date().toISOString();
    this.completedRuns = 0;
    this.totalRuns = plan.estimatedRuns;
    this.currentCandidate = null;
    this.lastResult = null;

    void this.execute(profile, plan).catch((error) => {
      this.lastResult = {
        sweepId: this.sweepId ?? "unknown",
        status: "failed",
        profileId: profile.id,
        profileName: profile.name,
        ranked: [],
        winnerRunId: null,
        baselineRunId: null,
        bestSettings: {},
        notes: [error instanceof Error ? error.message : String(error)]
      };
      this.running = false;
      this.currentCandidate = null;
    });

    return this.getStatus();
  }

  async stop(): Promise<SweepStatus> {
    if (this.running) {
      this.cancelled = true;
      await this.benchmark.stop();
    }
    return this.getStatus();
  }

  private runOne(profile: LlamaProfile, settings: BenchmarkSettings, label: string): Promise<BenchmarkRun> {
    return new Promise((resolve, reject) => {
      this.benchmark
        .start(profile, settings, {
          sweepId: this.sweepId ?? undefined,
          sweepLabel: label,
          onComplete: resolve
        })
        .catch(reject);
    });
  }

  private async execute(profile: LlamaProfile, plan: SweepPlan): Promise<void> {
    const sweepId = this.sweepId!;
    const notes: string[] = [...plan.notes];
    const allRuns: BenchmarkRun[] = [];
    const seen = new Map<string, BenchmarkRun>();
    const maxGpuLayers = plan.maxGpuLayers ?? null;

    let bestOverrides: Partial<LlamaProfile> = {};
    let bestSettingOverrides: Partial<BenchmarkSettings> = {};
    let bestRun: BenchmarkRun | null = null;
    let bestSigma = 0;
    let status: SweepResult["status"] = "completed";

    outer: for (const stage of plan.stages) {
      const stageResults: Array<{
        run: BenchmarkRun;
        overrides: Partial<LlamaProfile>;
        settingOverrides: Partial<BenchmarkSettings>;
      }> = [];

      for (const candidate of stage.candidates) {
        if (candidate.prunedReason) {
          continue;
        }
        if (this.cancelled) {
          status = "cancelled";
          break outer;
        }

        const mergedProfile = normalizeProfile({ ...profile, ...bestOverrides, ...candidate.profileOverrides });
        const mergedSettings: BenchmarkSettings = {
          ...plan.benchSettings,
          ...bestSettingOverrides,
          ...candidate.settings
        };
        if (mergedSettings.ubatchSize > mergedSettings.batchSize) {
          notes.push(`Skipped "${candidate.label}": micro-batch ${mergedSettings.ubatchSize} exceeds batch ${mergedSettings.batchSize}.`);
          this.totalRuns = Math.max(0, this.totalRuns - 1);
          continue;
        }

        const key = configKey(mergedProfile, mergedSettings, maxGpuLayers);
        const existing = seen.get(key);
        if (existing) {
          // Identical effective config already measured (e.g. the baseline);
          // reuse its result instead of burning another run.
          stageResults.push({ run: existing, overrides: candidate.profileOverrides, settingOverrides: candidate.settings });
          this.totalRuns = Math.max(0, this.totalRuns - 1);
          continue;
        }

        this.currentCandidate = candidate.label;
        let run: BenchmarkRun;
        try {
          run = await this.runOne(mergedProfile, mergedSettings, candidate.label);
        } catch (error) {
          notes.push(`"${candidate.label}" failed to start: ${error instanceof Error ? error.message : String(error)}`);
          if (stage.axis === "baseline") {
            status = "failed";
            break outer;
          }
          continue;
        }
        this.completedRuns += 1;

        if (run.status === "cancelled") {
          status = "cancelled";
          break outer;
        }
        if (run.status !== "completed" || run.metrics.score === null) {
          notes.push(`"${candidate.label}" ${run.status === "completed" ? "produced no score" : run.status}${run.error ? `: ${run.error}` : "."}`);
          if (stage.axis === "baseline") {
            status = "failed";
            notes.push("The baseline run failed, so there is nothing to compare against.");
            break outer;
          }
          continue;
        }

        seen.set(key, run);
        allRuns.push(run);
        stageResults.push({ run, overrides: candidate.profileOverrides, settingOverrides: candidate.settings });

        if (this.delayMs > 0 && !this.cancelled) {
          await sleep(this.delayMs);
        }
      }

      // Carry the stage winner forward. A challenger must beat the incumbent
      // by more than combined 1σ noise — otherwise stick with fewer changes.
      for (const result of stageResults) {
        const score = result.run.metrics.score;
        if (score === null) {
          continue;
        }
        const sigma = scoreSigma(result.run) ?? score * 0.02;
        const incumbentScore = bestRun?.metrics.score ?? null;
        if (
          incumbentScore === null ||
          score - incumbentScore > Math.sqrt(sigma * sigma + bestSigma * bestSigma)
        ) {
          bestRun = result.run;
          bestSigma = sigma;
          bestOverrides = { ...bestOverrides, ...result.overrides };
          bestSettingOverrides = { ...bestSettingOverrides, ...result.settingOverrides };
        }
      }
    }

    const ranking = rankSweep(allRuns);
    const winnerRun = allRuns.find((run) => run.id === ranking.winnerRunId) ?? null;
    this.lastResult = {
      sweepId,
      status,
      profileId: profile.id,
      profileName: profile.name,
      ranked: ranking.ranked,
      winnerRunId: ranking.winnerRunId,
      baselineRunId: ranking.baselineRunId,
      bestSettings: winnerRun ? deriveBestSettings(winnerRun) : {},
      notes: [...notes, ...ranking.notes]
    };
    this.running = false;
    this.currentCandidate = null;
  }
}
