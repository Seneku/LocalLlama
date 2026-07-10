import { describe, expect, test } from "bun:test";

import { formatModelType, formatParamCount } from "../src/shared/modelLabel";
import type { BenchmarkEnv } from "../src/shared/types";

function env(overrides: Partial<BenchmarkEnv>): BenchmarkEnv {
  return {
    buildNumber: null,
    buildCommit: null,
    gpuName: null,
    cpuName: null,
    backends: null,
    modelType: null,
    modelSizeBytes: null,
    modelParams: null,
    nGpuLayers: null,
    flashAttn: null,
    ...overrides
  };
}

describe("formatParamCount", () => {
  test("rounds billions to whole numbers at 10B+ and one decimal below", () => {
    expect(formatParamCount(11_675_250_624)).toBe("12B"); // the reported 11.7B case
    expect(formatParamCount(69_000_000_000)).toBe("69B");
    expect(formatParamCount(7_600_000_000)).toBe("7.6B");
    expect(formatParamCount(3_000_000_000)).toBe("3B");
  });

  test("shows millions below 1B", () => {
    expect(formatParamCount(494_000_000)).toBe("494M");
  });
});

describe("formatModelType", () => {
  test("replaces llama.cpp's wrong size bucket with the true parameter count", () => {
    // The exact case from the user's benchmarks: an 11.7B model bucketed as "2B".
    expect(formatModelType(env({ modelType: "qwen35 2B Q4_K - Medium", modelParams: 11_675_250_624 }))).toBe(
      "qwen35 12B Q4_K - Medium"
    );
  });

  test("fixes the unknown-size '?B' bucket", () => {
    expect(formatModelType(env({ modelType: "gemma4 ?B Q4_K - Medium", modelParams: 12_200_000_000 }))).toBe(
      "gemma4 12B Q4_K - Medium"
    );
  });

  test("corrects MoE-style buckets", () => {
    expect(formatModelType(env({ modelType: "llama4 8x7B Q4_K - Medium", modelParams: 47_000_000_000 }))).toBe(
      "llama4 47B Q4_K - Medium"
    );
  });

  test("keeps the string unchanged when params are unavailable", () => {
    expect(formatModelType(env({ modelType: "qwen35 2B Q4_K - Medium", modelParams: null }))).toBe(
      "qwen35 2B Q4_K - Medium"
    );
  });

  test("annotates rather than corrupts an unexpected shape", () => {
    expect(formatModelType(env({ modelType: "some-oddball-label", modelParams: 8_000_000_000 }))).toBe(
      "some-oddball-label (8B)"
    );
  });

  test("returns null when there is no model_type", () => {
    expect(formatModelType(env({ modelType: null, modelParams: 8e9 }))).toBeNull();
    expect(formatModelType(null)).toBeNull();
  });
});
