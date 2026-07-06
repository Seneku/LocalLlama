import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import { createBenchmarkStore, type BenchmarkStore } from "./benchmarkStore";
import { getDefaultThreads, getRuntimePaths, type RuntimePaths } from "./paths";
import { killTree } from "./runtime";
import type {
  BenchmarkCommandPreview,
  BenchmarkMetrics,
  BenchmarkProfileSnapshot,
  BenchmarkRow,
  BenchmarkRun,
  BenchmarkSettings,
  BenchmarkStatus,
  LlamaProfile,
  LogStream,
  ResolvedBackend,
  RuntimeLog
} from "../src/shared/types";

const MAX_BENCHMARK_LOGS = 500;
// Live stderr is only used for surfacing an error tail, so keep the last 64 KB.
const MAX_STDERR_BYTES = 64 * 1024;
// stdout must retain the full trailing JSON array, so keep a generous 1 MB tail.
const MAX_STDOUT_BYTES = 1024 * 1024;
// Logs are captured separately, so persist only a small tail on the stored run.
const STORED_STREAM_BYTES = 64 * 1024;

function capTail(value: string, maxBytes: number): string {
  return value.length > maxBytes ? value.slice(value.length - maxBytes) : value;
}

export const defaultBenchmarkSettings: BenchmarkSettings = {
  promptTokens: 512,
  generationTokens: 128,
  repetitions: 3,
  batchSize: 2048,
  ubatchSize: 512,
  noWarmup: false,
  flashAttention: "auto"
};

interface BuildBenchmarkCommandOptions {
  paths?: RuntimePaths;
  defaultThreads?: number;
  fileExists?: (filePath: string) => boolean;
}

function positiveInteger(value: number, fallback: number): number {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return Math.max(1, Math.floor(value));
}

function quoteArg(arg: string): string {
  if (arg.length === 0) {
    return "\"\"";
  }
  if (!/[\s"]/u.test(arg)) {
    return arg;
  }
  return `"${arg.replaceAll("\"", "\\\"")}"`;
}

function resolveBackend(
  mode: LlamaProfile["backendMode"],
  paths: RuntimePaths,
  fileExists: (filePath: string) => boolean
): ResolvedBackend {
  if (mode === "cuda") {
    return "CUDA";
  }
  if (mode === "cpu") {
    return "CPU";
  }
  return fileExists(paths.cudaBenchPath) ? "CUDA" : "CPU";
}

function normalizedSettings(settings?: Partial<BenchmarkSettings>): BenchmarkSettings {
  return {
    promptTokens: positiveInteger(settings?.promptTokens ?? defaultBenchmarkSettings.promptTokens, 512),
    generationTokens: positiveInteger(settings?.generationTokens ?? defaultBenchmarkSettings.generationTokens, 128),
    repetitions: positiveInteger(settings?.repetitions ?? defaultBenchmarkSettings.repetitions, 3),
    batchSize: positiveInteger(settings?.batchSize ?? defaultBenchmarkSettings.batchSize, 2048),
    ubatchSize: positiveInteger(settings?.ubatchSize ?? defaultBenchmarkSettings.ubatchSize, 512),
    noWarmup: Boolean(settings?.noWarmup ?? defaultBenchmarkSettings.noWarmup),
    flashAttention: settings?.flashAttention ?? defaultBenchmarkSettings.flashAttention
  };
}

export function buildBenchmarkCommand(
  profile: LlamaProfile,
  settings?: Partial<BenchmarkSettings>,
  options: BuildBenchmarkCommandOptions = {}
): BenchmarkCommandPreview {
  const paths = options.paths ?? getRuntimePaths();
  const defaultThreads = options.defaultThreads ?? getDefaultThreads();
  const fileExists = options.fileExists ?? fs.existsSync;
  const backend = resolveBackend(profile.backendMode, paths, fileExists);
  const executable = backend === "CUDA" ? paths.cudaBenchPath : paths.cpuBenchPath;
  const benchSettings = normalizedSettings(settings);
  const threads =
    profile.threadsMode === "manual" ? positiveInteger(profile.threads, defaultThreads) : defaultThreads;
  const warnings: string[] = [];
  const args = [
    "-o",
    "json",
    "-m",
    profile.modelPath,
    "-p",
    String(benchSettings.promptTokens),
    "-n",
    String(benchSettings.generationTokens),
    "-r",
    String(benchSettings.repetitions),
    "-b",
    String(benchSettings.batchSize),
    "-ub",
    String(benchSettings.ubatchSize),
    "-t",
    String(threads),
    "-fa",
    benchSettings.flashAttention
  ];

  if (benchSettings.noWarmup) {
    args.push("--no-warmup");
  }
  if (profile.kvCacheK) {
    args.push("-ctk", profile.kvCacheK);
  }
  if (profile.kvCacheV) {
    args.push("-ctv", profile.kvCacheV);
  }
  if (backend === "CUDA" && profile.gpuLayers > 0) {
    args.push("-ngl", String(Math.floor(profile.gpuLayers)));
  }
  if (backend === "CPU" && profile.gpuLayers > 0) {
    warnings.push("GPU layers are ignored when the CPU benchmark backend is selected.");
  }
  if (profile.jinja || profile.reasoning !== "off" || profile.parallelSlots !== 1) {
    warnings.push("Server-only settings such as Jinja, reasoning, host/port, and parallel slots are not used by llama-bench.");
  }
  if (profile.speculative.enabled) {
    warnings.push("llama-bench does not expose speculative/MTP server decoding flags, so this run measures base model throughput.");
  }

  return {
    executable,
    args,
    display: [executable, ...args].map(quoteArg).join(" "),
    backend,
    benchmarkExists: fileExists(executable),
    modelExists: fileExists(profile.modelPath),
    warnings
  };
}

export function validateBenchmarkLaunch(profile: LlamaProfile, settings?: Partial<BenchmarkSettings>) {
  const command = buildBenchmarkCommand(profile, settings);
  const errors: string[] = [];

  if (!command.benchmarkExists) {
    errors.push(`llama-bench not found: ${command.executable}`);
  }
  if (!command.modelExists) {
    errors.push(`model not found: ${profile.modelPath}`);
  }
  return { command, errors };
}

function textValue(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function numberValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function extractJson(stdout: string): unknown {
  const arrayStart = stdout.indexOf("[");
  const arrayEnd = stdout.lastIndexOf("]");
  if (arrayStart >= 0 && arrayEnd > arrayStart) {
    return JSON.parse(stdout.slice(arrayStart, arrayEnd + 1));
  }
  const objectStart = stdout.indexOf("{");
  const objectEnd = stdout.lastIndexOf("}");
  if (objectStart >= 0 && objectEnd > objectStart) {
    return JSON.parse(stdout.slice(objectStart, objectEnd + 1));
  }
  throw new Error("benchmark output did not contain JSON results");
}

export function parseBenchmarkRows(stdout: string): BenchmarkRow[] {
  const parsed = extractJson(stdout);
  const rows = Array.isArray(parsed) ? parsed : [parsed];

  return rows.map((row) => {
    const raw = row && typeof row === "object" ? (row as Record<string, unknown>) : {};
    const promptTokens = numberValue(raw.n_prompt ?? raw.prompt_tokens) ?? 0;
    const generationTokens = numberValue(raw.n_gen ?? raw.generation_tokens) ?? 0;
    const avgTokensPerSecond = numberValue(raw.avg_ts ?? raw.tokens_per_second ?? raw.tps);
    const stddevTokensPerSecond = numberValue(raw.stddev_ts ?? raw.tokens_per_second_stddev);
    const avgNs = numberValue(raw.avg_ns);
    const avgMilliseconds = avgNs === null ? numberValue(raw.avg_ms) : avgNs / 1_000_000;

    return {
      test: textValue(raw.test, promptTokens > 0 && generationTokens === 0 ? `pp${promptTokens}` : `tg${generationTokens}`),
      promptTokens,
      generationTokens,
      avgTokensPerSecond,
      stddevTokensPerSecond,
      avgMilliseconds,
      raw
    };
  });
}

export function calculateBenchmarkMetrics(rows: BenchmarkRow[], startedAt: string, completedAt: string | null): BenchmarkMetrics {
  const promptRow =
    rows.find((row) => row.test.toLowerCase().startsWith("pp") && row.avgTokensPerSecond !== null) ??
    rows.find((row) => row.promptTokens > 0 && row.generationTokens === 0 && row.avgTokensPerSecond !== null) ??
    null;
  const generationRow =
    rows.find((row) => row.test.toLowerCase().startsWith("tg") && row.avgTokensPerSecond !== null) ??
    rows.find((row) => row.generationTokens > 0 && row.avgTokensPerSecond !== null) ??
    null;
  const promptTokensPerSecond = promptRow?.avgTokensPerSecond ?? null;
  const generationTokensPerSecond = generationRow?.avgTokensPerSecond ?? null;
  const totalSeconds = completedAt
    ? Math.max(0, (Date.parse(completedAt) - Date.parse(startedAt)) / 1000)
    : null;

  return {
    promptTokensPerSecond,
    generationTokensPerSecond,
    generationMsPerToken: generationTokensPerSecond ? 1000 / generationTokensPerSecond : null,
    promptStddev: promptRow?.stddevTokensPerSecond ?? null,
    generationStddev: generationRow?.stddevTokensPerSecond ?? null,
    totalSeconds,
    score:
      promptTokensPerSecond !== null && generationTokensPerSecond !== null
        ? Math.sqrt(promptTokensPerSecond * generationTokensPerSecond)
        : promptTokensPerSecond ?? generationTokensPerSecond
  };
}

function createProfileSnapshot(profile: LlamaProfile): BenchmarkProfileSnapshot {
  return {
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
  };
}

function createRunId(profileId: string): string {
  const suffix = Math.random().toString(36).slice(2, 8);
  return `${profileId}-${Date.now().toString(36)}-${suffix}`;
}

export class BenchmarkManager {
  private process: ChildProcessWithoutNullStreams | null = null;
  private activeRun: BenchmarkRun | null = null;
  private stopRequested = false;
  private logs: RuntimeLog[] = [];
  private nextLogId = 1;
  private buffers: Record<"stdout" | "stderr", string> = { stdout: "", stderr: "" };
  private stdout = "";
  private stderr = "";

  constructor(private readonly store: BenchmarkStore = createBenchmarkStore()) {}

  getStatus(): BenchmarkStatus {
    return {
      state: this.process ? "running" : "idle",
      activeRunId: this.activeRun?.id ?? null,
      profileId: this.activeRun?.profileId ?? null,
      profileName: this.activeRun?.profileName ?? null,
      startedAt: this.activeRun?.createdAt ?? null,
      command: this.activeRun?.command ?? null
    };
  }

  getLogs(): RuntimeLog[] {
    return this.logs;
  }

  async start(profile: LlamaProfile, settings?: Partial<BenchmarkSettings>): Promise<BenchmarkRun> {
    if (this.process) {
      throw new Error(`A benchmark is already running for ${this.activeRun?.profileName ?? "another profile"}.`);
    }

    const normalized = normalizedSettings(settings);
    const validation = validateBenchmarkLaunch(profile, normalized);
    if (validation.errors.length > 0) {
      throw new Error(validation.errors.join("\n"));
    }

    this.stopRequested = false;
    this.logs = [];
    this.stdout = "";
    this.stderr = "";
    this.buffers = { stdout: "", stderr: "" };

    const run: BenchmarkRun = {
      id: createRunId(profile.id),
      profileId: profile.id,
      profileName: profile.name,
      createdAt: new Date().toISOString(),
      completedAt: null,
      status: "running",
      exitCode: null,
      signal: null,
      backend: validation.command.backend,
      settings: normalized,
      command: validation.command,
      profile: createProfileSnapshot(profile),
      rows: [],
      metrics: calculateBenchmarkMetrics([], new Date().toISOString(), null),
      stdout: "",
      stderr: "",
      error: null
    };

    this.activeRun = run;
    await this.store.upsert(run);
    this.appendLog("system", `Starting benchmark for ${profile.name}`);
    this.appendLog("system", validation.command.display);

    this.process = spawn(validation.command.executable, validation.command.args, {
      cwd: path.dirname(validation.command.executable),
      shell: false,
      windowsHide: true
    });
    this.process.stdout.on("data", (chunk: Buffer) => {
      this.stdout = capTail(this.stdout + chunk.toString("utf8"), MAX_STDOUT_BYTES);
      this.appendChunk("stdout", chunk);
    });
    this.process.stderr.on("data", (chunk: Buffer) => {
      this.stderr = capTail(this.stderr + chunk.toString("utf8"), MAX_STDERR_BYTES);
      this.appendChunk("stderr", chunk);
    });
    this.process.once("error", async (error) => {
      await this.finishRun("failed", null, null, error.message);
    });
    this.process.once("exit", async (code, signal) => {
      const status = this.stopRequested ? "cancelled" : code === 0 ? "completed" : "failed";
      await this.finishRun(status, code, signal, null);
    });

    return run;
  }

  async stop(): Promise<BenchmarkStatus> {
    if (!this.process) {
      return this.getStatus();
    }
    this.stopRequested = true;
    const pid = this.process.pid;
    this.appendLog("system", `Stopping benchmark process ${pid ?? "unknown"}.`);
    this.process.kill();
    // On Windows plain kill() does not terminate the child process tree, so
    // always reap the whole tree via taskkill /T /F to release VRAM.
    if (process.platform === "win32" && pid) {
      await killTree(pid, (message) => this.appendLog("system", message));
    }
    return this.getStatus();
  }

  private async finishRun(
    status: BenchmarkRun["status"],
    code: number | null,
    signal: string | null,
    processError: string | null
  ): Promise<void> {
    if (!this.activeRun) {
      this.process = null;
      return;
    }

    this.flushBuffers();
    const completedAt = new Date().toISOString();
    let rows: BenchmarkRow[] = [];
    let error = processError;
    if (status === "completed") {
      try {
        rows = parseBenchmarkRows(this.stdout);
      } catch (parseError) {
        error = parseError instanceof Error ? parseError.message : String(parseError);
        status = "failed";
      }
    } else if (!error && status === "failed") {
      error = this.stderr.trim() || `benchmark exited with code ${code ?? "null"}`;
    }

    const completedRun: BenchmarkRun = {
      ...this.activeRun,
      completedAt,
      status,
      exitCode: code,
      signal,
      rows,
      metrics: calculateBenchmarkMetrics(rows, this.activeRun.createdAt, completedAt),
      stdout: capTail(this.stdout, STORED_STREAM_BYTES),
      stderr: capTail(this.stderr, STORED_STREAM_BYTES),
      error
    };
    this.activeRun = null;
    this.process = null;
    await this.store.upsert(completedRun);
    this.appendLog("system", `Benchmark ${status}.`);
  }

  private appendChunk(stream: "stdout" | "stderr", chunk: Buffer): void {
    const text = this.buffers[stream] + chunk.toString("utf8");
    const lines = text.split(/\r?\n/u);
    this.buffers[stream] = lines.pop() ?? "";
    for (const line of lines) {
      if (line.trim()) {
        this.appendLog(stream, line);
      }
    }
  }

  private flushBuffers(): void {
    for (const stream of ["stdout", "stderr"] as const) {
      const line = this.buffers[stream];
      if (line.trim()) {
        this.appendLog(stream, line);
      }
      this.buffers[stream] = "";
    }
  }

  private appendLog(stream: LogStream, line: string): void {
    this.logs.push({
      id: this.nextLogId++,
      time: new Date().toISOString(),
      stream,
      line
    });
    if (this.logs.length > MAX_BENCHMARK_LOGS) {
      this.logs.splice(0, this.logs.length - MAX_BENCHMARK_LOGS);
    }
  }
}
