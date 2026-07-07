import { describe, expect, test } from "bun:test";

import {
  buildBenchmarkCommand,
  calculateBenchmarkMetrics,
  defaultBenchmarkSettings,
  parseBenchmarkRows
} from "../server/benchmark";
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
