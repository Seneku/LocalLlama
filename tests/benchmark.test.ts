import { describe, expect, test } from "bun:test";

import {
  buildBenchmarkCommand,
  calculateBenchmarkMetrics,
  defaultBenchmarkSettings,
  parseBenchmarkRows
} from "../server/benchmark";
import { extractBenchmarkEnv } from "../server/benchmarkEnv";
import type { BenchmarkRow } from "../src/shared/types";
import { exampleProfiles } from "../server/defaultProfiles";
import type { RuntimePaths } from "../server/paths";

const paths: RuntimePaths = {
  llamaRoot: "C:\\llama.cpp",
  cudaServerPath: "C:\\llama.cpp\\dist-cuda\\llama-server.exe",
  cpuServerPath: "C:\\llama.cpp\\build\\bin\\llama-server.exe",
  cudaBenchPath: "C:\\llama.cpp\\dist-cuda\\llama-bench.exe",
  cpuBenchPath: "C:\\llama.cpp\\build\\bin\\llama-bench.exe",
  dataPath: "C:\\LocalLlama\\data"
};

const exists = (filePath: string) =>
  [paths.cudaBenchPath, paths.cpuBenchPath, ...exampleProfiles.map((profile) => profile.modelPath)].includes(filePath);

describe("benchmark command generation", () => {
  test("uses profile backend, model, threads, cache, and GPU settings", () => {
    const profile = exampleProfiles.find((item) => item.id === "orinth9b-mtp-coding")!;
    const preview = buildBenchmarkCommand(profile, defaultBenchmarkSettings, {
      paths,
      defaultThreads: 16,
      fileExists: exists
    });

    expect(preview.backend).toBe("CUDA");
    expect(preview.args).toContain("-m");
    expect(preview.args).toContain(profile.modelPath);
    expect(preview.args).toContain("-ngl");
    expect(preview.args).toContain("99");
    expect(preview.args).toContain("-ctk");
    expect(preview.args).toContain("q8_0");
    expect(preview.args).toContain("-t");
    expect(preview.args).toContain("16");
    expect(preview.warnings.some((warning) => warning.includes("speculative"))).toBe(true);
  });

  test("falls back to CPU benchmark when CUDA bench is missing", () => {
    const profile = exampleProfiles[0];
    const preview = buildBenchmarkCommand(profile, defaultBenchmarkSettings, {
      paths,
      defaultThreads: 8,
      fileExists: (filePath) => filePath !== paths.cudaBenchPath && exists(filePath)
    });

    expect(preview.backend).toBe("CPU");
    expect(preview.args).not.toContain("-ngl");
    expect(preview.warnings).toContain("GPU layers are ignored when the CPU benchmark backend is selected.");
  });

  test("seeds batch/ubatch/flash-attention from the profile when not overridden", () => {
    const profile = {
      ...exampleProfiles[0],
      batchSize: 1024,
      ubatchSize: 256,
      flashAttention: "on" as const
    };
    const fromProfile = buildBenchmarkCommand(profile, undefined, {
      paths,
      defaultThreads: 8,
      fileExists: exists
    }).args.join(" ");
    expect(fromProfile).toContain("-b 1024");
    expect(fromProfile).toContain("-ub 256");
    expect(fromProfile).toContain("-fa on");

    // Explicit benchmark settings still win over the profile.
    const overridden = buildBenchmarkCommand(profile, { batchSize: 512, flashAttention: "off" }, {
      paths,
      defaultThreads: 8,
      fileExists: exists
    }).args.join(" ");
    expect(overridden).toContain("-b 512");
    expect(overridden).toContain("-ub 256"); // still from the profile
    expect(overridden).toContain("-fa off");
  });
});

describe("benchmark parsing", () => {
  test("parses llama-bench json rows and key metrics", () => {
    const stdout = JSON.stringify([
      {
        test: "pp512",
        n_prompt: 512,
        n_gen: 0,
        avg_ts: 1122.45,
        stddev_ts: 12.1,
        avg_ns: 456000000
      },
      {
        test: "tg128",
        n_prompt: 0,
        n_gen: 128,
        avg_ts: 74.88,
        stddev_ts: 1.4,
        avg_ns: 1709000000
      }
    ]);

    const rows = parseBenchmarkRows(stdout);
    const metrics = calculateBenchmarkMetrics(rows, "2026-07-06T12:00:00.000Z", "2026-07-06T12:00:14.000Z");

    expect(rows).toHaveLength(2);
    expect(metrics.promptTokensPerSecond).toBe(1122.45);
    expect(metrics.generationTokensPerSecond).toBe(74.88);
    expect(metrics.generationMsPerToken).toBeCloseTo(13.3547, 4);
    expect(metrics.totalSeconds).toBe(14);
    expect(metrics.score).toBeCloseTo(Math.sqrt(1122.45 * 74.88), 4);
  });
});

describe("extractBenchmarkEnv", () => {
  function row(raw: Record<string, unknown>): BenchmarkRow {
    return {
      test: "pp512",
      promptTokens: 512,
      generationTokens: 0,
      avgTokensPerSecond: 1000,
      stddevTokensPerSecond: 10,
      avgMilliseconds: 500,
      raw
    };
  }

  test("hoists build, hardware, and model identity from a realistic raw row", () => {
    const env = extractBenchmarkEnv([
      row({
        build_commit: "e3ba22d6c",
        build_number: 9503,
        cpu_info: "AMD Ryzen 9 7950X 16-Core Processor",
        gpu_info: "NVIDIA GeForce RTX 4070 SUPER",
        backends: "CUDA",
        model_type: "qwen35 9B Q4_K - Medium",
        model_size: 5444231168,
        model_n_params: 9401247744,
        n_gpu_layers: 99,
        flash_attn: 1
      })
    ]);
    expect(env).toMatchObject({
      buildNumber: 9503,
      buildCommit: "e3ba22d6c",
      gpuName: "NVIDIA GeForce RTX 4070 SUPER",
      cpuName: "AMD Ryzen 9 7950X 16-Core Processor",
      backends: "CUDA",
      modelType: "qwen35 9B Q4_K - Medium",
      modelParams: 9401247744,
      nGpuLayers: 99,
      flashAttn: true
    });
  });

  test("missing fields become nulls, never throws", () => {
    const env = extractBenchmarkEnv([row({ avg_ts: 100 })]);
    expect(env).toEqual({
      buildNumber: null,
      buildCommit: null,
      gpuName: null,
      cpuName: null,
      backends: null,
      modelType: null,
      modelSizeBytes: null,
      modelParams: null,
      nGpuLayers: null,
      flashAttn: null
    });
  });

  test("empty rows or empty raw yield null", () => {
    expect(extractBenchmarkEnv([])).toBeNull();
    expect(extractBenchmarkEnv([row({})])).toBeNull();
  });

  test("flash_attn variants coerce across llama-bench builds", () => {
    expect(extractBenchmarkEnv([row({ flash_attn: true, x: 1 })])?.flashAttn).toBe(true);
    expect(extractBenchmarkEnv([row({ flash_attn: 0, x: 1 })])?.flashAttn).toBe(false);
    expect(extractBenchmarkEnv([row({ flash_attn: "1", x: 1 })])?.flashAttn).toBe(true);
    expect(extractBenchmarkEnv([row({ flash_attn: "false", x: 1 })])?.flashAttn).toBe(false);
  });
});
