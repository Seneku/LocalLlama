import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import { createBenchmarkStore, MAX_STORED_RUNS } from "../server/benchmarkStore";
import type { BenchmarkRun } from "../src/shared/types";

let tempDir = "";
let file = "";

beforeEach(() => {
  tempDir = mkdtempSync(path.join(tmpdir(), "localllama-bench-"));
  file = path.join(tempDir, "benchmarks.json");
});

afterEach(() => {
  rmSync(tempDir, { recursive: true, force: true });
});

function makeRun(id: string, extra: Partial<BenchmarkRun> = {}): BenchmarkRun {
  return {
    id,
    profileId: "p1",
    profileName: "Profile One",
    createdAt: new Date().toISOString(),
    completedAt: new Date().toISOString(),
    status: "completed",
    exitCode: 0,
    signal: null,
    backend: "CUDA",
    settings: {
      promptTokens: 512,
      generationTokens: 128,
      repetitions: 3,
      batchSize: 2048,
      ubatchSize: 512,
      noWarmup: false,
      flashAttention: "auto"
    },
    command: {
      executable: "llama-bench.exe",
      args: [],
      display: "llama-bench.exe",
      backend: "CUDA",
      benchmarkExists: true,
      modelExists: true,
      warnings: []
    },
    profile: {
      name: "Profile One",
      modelPath: "C:\\models\\test-Q4_K_M.gguf",
      backendMode: "auto",
      contextSize: 4096,
      threadsMode: "auto",
      threads: 8,
      gpuLayers: 99,
      kvCacheK: "",
      kvCacheV: "",
      jinja: false,
      reasoning: "auto",
      parallelSlots: 0,
      speculativeEnabled: false,
      speculativeType: "none"
    } as BenchmarkRun["profile"],
    rows: [],
    metrics: {
      promptTokensPerSecond: 1000,
      generationTokensPerSecond: 75,
      generationMsPerToken: 13.3,
      promptStddev: 10,
      generationStddev: 1,
      totalSeconds: 12,
      score: Math.sqrt(1000 * 75)
    },
    env: null,
    stdout: "",
    stderr: "",
    error: null,
    ...extra
  };
}

describe("benchmark store", () => {
  test("load backfills env on legacy runs missing the field", async () => {
    // Simulate a pre-enrichment file: runs without any `env` key but with raw rows.
    const legacy = makeRun("legacy-1", {
      rows: [
        {
          test: "pp512",
          promptTokens: 512,
          generationTokens: 0,
          avgTokensPerSecond: 1000,
          stddevTokensPerSecond: 5,
          avgMilliseconds: 500,
          raw: { build_number: 9503, gpu_info: "RTX 4070 SUPER", model_type: "9B Q4_K - Medium" }
        }
      ]
    });
    delete (legacy as unknown as Record<string, unknown>).env;
    writeFileSync(file, JSON.stringify([legacy], null, 2), "utf8");

    const store = createBenchmarkStore(file);
    const runs = await store.load();
    expect(runs[0].env).toMatchObject({ buildNumber: 9503, gpuName: "RTX 4070 SUPER", modelType: "9B Q4_K - Medium" });

    // Migration is persisted (through the write chain) — allow it to flush.
    await store.save(runs);
    const onDisk = JSON.parse(readFileSync(file, "utf8")) as BenchmarkRun[];
    expect(onDisk[0].env).toMatchObject({ buildNumber: 9503 });
  });

  test("legacy run without rows backfills env: null", async () => {
    const legacy = makeRun("legacy-2", { rows: [], status: "failed" });
    delete (legacy as unknown as Record<string, unknown>).env;
    writeFileSync(file, JSON.stringify([legacy], null, 2), "utf8");

    const store = createBenchmarkStore(file);
    const runs = await store.load();
    expect(runs[0].env).toBeNull();
  });

  test("upsert caps history at MAX_STORED_RUNS", async () => {
    const seed = Array.from({ length: MAX_STORED_RUNS }, (_, index) => makeRun(`run-${index}`));
    writeFileSync(file, JSON.stringify(seed), "utf8");

    const store = createBenchmarkStore(file);
    await store.upsert(makeRun("newest"));
    const runs = await store.load();
    expect(runs).toHaveLength(MAX_STORED_RUNS);
    expect(runs[0].id).toBe("newest");
    expect(runs.some((run) => run.id === `run-${MAX_STORED_RUNS - 1}`)).toBe(false);
  });

  test("delete removes by id", async () => {
    const store = createBenchmarkStore(file);
    await store.upsert(makeRun("a"));
    await store.upsert(makeRun("b"));
    expect(await store.delete("a")).toBe(true);
    expect(await store.delete("a")).toBe(false);
    const runs = await store.load();
    expect(runs.map((run) => run.id)).toEqual(["b"]);
  });
});
